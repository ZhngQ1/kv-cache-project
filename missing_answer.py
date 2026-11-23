"""Missing-answer control for the single-needle task, with one cache setting per process.

Every trial produces two prompts that share the same filler and the same NOT FOUND instruction:
  absent   no verification code anywhere; the correct output is exactly NOT FOUND
  present  the needle is inserted at --depth; the correct output is the code
Absent cases measure abstention. Present cases check that the instruction does not make the
model abstain when the answer is in the prompt. The two are scored and reported separately.
"""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import platform
import random
import re
import sys

from transformers import AutoTokenizer

from needle import GEN_TOKENS, MARKER, cpu_runner, exact_match, vllm_runner

TASK = "missing-answer-v1"
ABSTAIN = "NOT FOUND"
QUESTION = (
    "Question: What is the archive's verification code? "
    "If the archive does not contain a verification code, reply NOT FOUND. "
    "Reply with only the code or NOT FOUND."
)
SIX_DIGITS = re.compile(r"(?<!\d)\d{6}(?!\d)")
VARIANTS = ("absent", "present")
OUTCOMES = ("abstain", "target_code", "other_code", "other")


def check_prompt(text, variant, code):
    """Absent prompts must hold no code; present prompts must hold exactly the target code."""
    numbers = SIX_DIGITS.findall(text)
    statements = text.count("verification code is")
    if variant == "absent" and (numbers or statements):
        raise ValueError(f"Absent prompt contains a code: {numbers}")
    if variant == "present" and (numbers != [code] or statements != 1):
        raise ValueError(f"Present prompt must contain only {code}, found {numbers}")


def cases(tokenizer, lengths, trials, seed, depth):
    message = f"Read the archive below.\n{MARKER}\n{QUESTION}"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True
    )
    left, right = rendered.split(MARKER)
    prefix = tokenizer.encode(left, add_special_tokens=False)
    suffix = tokenizer.encode(right, add_special_tokens=False)
    for length in lengths:
        for trial in range(trials):
            # Same random draws as needle.py: the code first, then the filler notes.
            rng = random.Random(seed + trial)
            code = str(rng.randrange(100000, 1000000))
            needle = tokenizer.encode(
                f"\nThe archive's verification code is {code}.\n", add_special_tokens=False
            )
            absent_count = length - len(prefix) - len(suffix)
            present_count = absent_count - len(needle)
            if present_count < 1:
                raise ValueError(f"Prompt length {length} is too small for this tokenizer.")
            filler = []
            while len(filler) < absent_count:
                note = (
                    f"\nIn the {rng.choice(['north', 'east', 'west', 'central'])} district, "
                    f"the {rng.choice(['library', 'clinic', 'museum', 'station'])} "
                    f"received {rng.randint(10, 99)} visitors on "
                    f"{rng.choice(['Monday', 'Tuesday', 'Wednesday', 'Friday'])}."
                )
                filler.extend(tokenizer.encode(note, add_special_tokens=False))
            # The present filler is the absent filler minus its last len(needle) tokens,
            # so both prompts have exactly `length` tokens and share the same archive text.
            present_filler = filler[:present_count]
            offset = round(depth * present_count)
            start = len(prefix) + offset
            prompts = {
                "absent": (prefix + filler[:absent_count] + suffix, ABSTAIN, None, None, None),
                "present": (
                    prefix + present_filler[:offset] + needle + present_filler[offset:] + suffix,
                    code, depth, start, start + len(needle),
                ),
            }
            for variant in VARIANTS:
                ids, expected, case_depth, needle_start, needle_end = prompts[variant]
                if len(ids) != length:
                    raise ValueError(f"{variant} prompt has {len(ids)} tokens, expected {length}")
                check_prompt(tokenizer.decode(ids), variant, code)
                yield ids, {
                    "task": TASK, "variant": variant, "seq_len": length, "trial": trial,
                    "depth": case_depth, "expected": expected,
                    "needle_start_token": needle_start, "needle_end_token_exclusive": needle_end,
                    "prompt_sha256": sha256(json.dumps(ids).encode()).hexdigest(),
                }


def classify(answer, code):
    """abstain: NOT FOUND in any letter case (optional final period); target_code: exactly the code;
    other_code: some other six-digit number; other: anything else."""
    text = answer.strip()
    if re.fullmatch(re.escape(ABSTAIN) + r"\.?", text, flags=re.IGNORECASE):
        return "abstain"
    if code is not None and exact_match(answer, code):
        return "target_code"
    if SIX_DIGITS.search(text):
        return "other_code"
    return "other"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", required=True, help="Hub commit SHA or local checkpoint identifier")
    parser.add_argument("--cache", choices=["fp32", "int8", "auto", "fp8_e4m3"],
                        help="Cache setting to run; omit to only write prompts with --dump-prompts")
    parser.add_argument("--lengths", type=int, nargs="+", default=[512])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--depth", type=float, default=0.5, help="Needle position for present cases")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=226)
    parser.add_argument("--gpu-dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--output", type=Path, help="Per-case results (JSONL), required with --cache")
    parser.add_argument("--dump-prompts", type=Path, help="Write token IDs and decoded text of every prompt")
    args = parser.parse_args()
    if min(*args.lengths, args.trials, args.threads) < 1:
        parser.error("lengths, trials and threads must be positive")
    if len(set(args.lengths)) != len(args.lengths):
        parser.error("lengths must be unique")
    if not 0 <= args.depth <= 1:
        parser.error("depth must be between 0 and 1")
    if args.cache is None and args.dump_prompts is None:
        parser.error("give --cache to run a model, --dump-prompts to write prompts, or both")
    if args.cache is not None and args.output is None:
        parser.error("--output is required with --cache")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # Build and check every prompt before loading a model, so fixture errors fail fast.
    all_cases = list(cases(tokenizer, args.lengths, args.trials, args.seed, args.depth))
    if args.dump_prompts:
        with args.dump_prompts.open("x") as dump:
            for ids, case in all_cases:
                print(json.dumps(dict(**case, token_ids=ids, text=tokenizer.decode(ids))), file=dump)
        print(f"wrote {len(all_cases)} prompts to {args.dump_prompts}", file=sys.stderr)
    if args.cache is None:
        return

    if args.cache in ("fp32", "int8"):
        run, metadata = cpu_runner(args, tokenizer)
        if max(args.lengths) + GEN_TOKENS > metadata["context_limit"]:
            parser.error("prompt plus generation budget exceeds model context limit")
    else:
        run, metadata = vllm_runner(args)
    counts = {variant: Counter() for variant in VARIANTS}
    with args.output.open("x") as output:
        for ids, case in all_cases:
            answer, memory = run(ids)
            code = case["expected"] if case["variant"] == "present" else None
            outcome = classify(answer, code)
            correct = outcome == ("abstain" if case["variant"] == "absent" else "target_code")
            counts[case["variant"]][outcome] += 1
            print(json.dumps(dict(
                **case, **metadata, **memory, answer=answer, outcome=outcome, correct=correct,
                cache=args.cache, model=args.model, revision=args.revision, seed=args.seed,
                max_new_tokens=GEN_TOKENS,
                os=platform.platform(),
            )), file=output, flush=True)
    for variant in VARIANTS:
        breakdown = ", ".join(f"{name} {counts[variant][name]}" for name in OUTCOMES)
        print(f"{args.cache} {variant}: {breakdown}", file=sys.stderr)


if __name__ == "__main__":
    main()
