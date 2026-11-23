"""Run token-matched single-needle retrieval with one cache setting per process."""
import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import re
import sys

from transformers import AutoTokenizer

MARKER = "HAYSTACK_PLACEHOLDER"
DEPTHS = (0.1, 0.5, 0.9)
GEN_TOKENS = 16


def cases(tokenizer, lengths, trials, seed, depths=DEPTHS):
    message = (
        "Read the archive below.\n"
        f"{MARKER}\n"
        "Question: What is the archive's verification code? Reply with only the code."
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )
    left, right = rendered.split(MARKER)
    prefix = tokenizer.encode(left, add_special_tokens=False)
    suffix = tokenizer.encode(right, add_special_tokens=False)
    for length in lengths:
        for trial in range(trials):
            rng = random.Random(seed + trial)
            code = str(rng.randrange(100000, 1000000))
            needle = tokenizer.encode(
                f"\nThe archive's verification code is {code}.\n", add_special_tokens=False
            )
            filler_count = length - len(prefix) - len(suffix) - len(needle)
            if filler_count < 1:
                raise ValueError(f"Prompt length {length} is too small for this tokenizer.")
            filler = []
            while len(filler) < filler_count:
                note = (
                    f"\nIn the {rng.choice(['north', 'east', 'west', 'central'])} district, "
                    f"the {rng.choice(['library', 'clinic', 'museum', 'station'])} "
                    f"received {rng.randint(10, 99)} visitors on "
                    f"{rng.choice(['Monday', 'Tuesday', 'Wednesday', 'Friday'])}."
                )
                filler.extend(tokenizer.encode(note, add_special_tokens=False))
            filler = filler[:filler_count]
            for depth in depths:
                offset = round(depth * filler_count)
                ids = prefix + filler[:offset] + needle + filler[offset:] + suffix
                assert len(ids) == length
                yield ids, {
                    "seq_len": length, "trial": trial, "depth": depth, "expected": code,
                    "needle_start_token": len(prefix) + offset,
                    "needle_end_token_exclusive": len(prefix) + offset + len(needle),
                    "prompt_sha256": sha256(json.dumps(ids).encode()).hexdigest(),
                }


def exact_match(answer, expected):
    return re.fullmatch(re.escape(expected) + r"\.?", answer.strip()) is not None


def cpu_runner(args, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache
    from baseline import payload_bytes
    from int8_cache import int8_cache, scale_bytes

    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32
    ).eval()
    if model.config.model_type != "qwen2" or model.config.use_sliding_window:
        raise ValueError("CPU runner requires a full-attention Qwen2 checkpoint.")
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos])

    @torch.inference_mode()
    def run(ids):
        cache = (
            int8_cache(model.config.num_hidden_layers)
            if args.cache == "int8" else DynamicCache(config=model.config)
        )
        tokens = torch.tensor([ids])
        emitted = []
        size = scales = 0
        for step in range(GEN_TOKENS):
            result = model(input_ids=tokens, past_key_values=cache, use_cache=True, logits_to_keep=1)
            if step == 0:
                size = payload_bytes(cache)
                scales = scale_bytes(cache) if args.cache == "int8" else 0
            tokens = result.logits.argmax(dim=-1)
            next_token = tokens.item()
            if next_token in stop_ids:
                break
            emitted.append(next_token)
        return tokenizer.decode(emitted, skip_special_tokens=True), {
            "prefill_kv_bytes": size, "prefill_scale_bytes": scales,
            "prefill_total_cache_bytes": size + scales,
        }

    return run, {
        "backend": "transformers", "weight_dtype": "float32", "device": "cpu",
        "torch": torch.__version__, "transformers": version("transformers"),
        "resolved_revision": model.config._commit_hash, "threads": args.threads,
        "attention_backend": model.config._attn_implementation,
        "context_limit": model.config.max_position_embeddings,
    }


def vllm_runner(args):
    # Match the validated reference runs (V0 engine, XFormers) unless the caller overrides them.
    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
    import torch
    from vllm import LLM, SamplingParams

    if not torch.cuda.is_available():
        raise RuntimeError("FP8 experiment requires a launch-tested CUDA or ROCm GPU backend.")
    fp8 = args.cache == "fp8_e4m3"
    engine = LLM(
        model=args.model, revision=args.revision, dtype=args.gpu_dtype,
        kv_cache_dtype=args.cache, calculate_kv_scales=fp8, enforce_eager=True,
        max_model_len=max(args.lengths) + GEN_TOKENS, max_num_seqs=1,
        enable_prefix_caching=False,
    )
    params = SamplingParams(temperature=0, max_tokens=GEN_TOKENS, seed=args.seed)

    def run(ids):
        result = engine.generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        assert result.prompt_token_ids == ids
        return result.outputs[0].text, {}

    return run, {
        "backend": "vllm", "vllm": version("vllm"), "torch": torch.__version__,
        "device": "rocm" if torch.version.hip else "cuda", "gpu": torch.cuda.get_device_name(),
        "cuda_version": torch.version.cuda, "hip_version": torch.version.hip,
        "weight_dtype": args.gpu_dtype, "calculate_kv_scales": fp8,
        "scale_policy": "dynamic" if fp8 else "native",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", required=True, help="Hub commit SHA or local checkpoint identifier")
    parser.add_argument("--cache", choices=["fp32", "int8", "auto", "fp8_e4m3"], required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--depths", type=float, nargs="+", default=list(DEPTHS),
                        help="Needle positions as fractions of the filler tokens")
    parser.add_argument("--task", help="Optional experiment label; stored in every record only when given")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=226)
    parser.add_argument("--gpu-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.lengths, args.trials, args.threads) < 1:
        parser.error("lengths, trials and threads must be positive")
    if len(set(args.lengths)) != len(args.lengths):
        parser.error("lengths must be unique")
    if len(set(args.depths)) != len(args.depths) or not all(0 <= d <= 1 for d in args.depths):
        parser.error("depths must be unique values between 0 and 1")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if args.cache in ("fp32", "int8"):
        run, metadata = cpu_runner(args, tokenizer)
        if max(args.lengths) + GEN_TOKENS > metadata["context_limit"]:
            parser.error("prompt plus generation budget exceeds model context limit")
    else:
        run, metadata = vllm_runner(args)
    correct = count = 0
    with args.output.open("x") as output:
        for ids, case in cases(tokenizer, args.lengths, args.trials, args.seed, args.depths):
            answer, memory = run(ids)
            hit = exact_match(answer, case["expected"])
            correct += hit
            count += 1
            record = dict(
                **case, **metadata, **memory, answer=answer, correct=hit, cache=args.cache,
                model=args.model, revision=args.revision, seed=args.seed, max_new_tokens=GEN_TOKENS,
                os=platform.platform(),
            )
            if args.task is not None:
                record = {"task": args.task, **record}
            print(json.dumps(record), file=output, flush=True)
    print(f"{args.cache}: {correct}/{count} exact matches", file=sys.stderr)


if __name__ == "__main__":
    main()
