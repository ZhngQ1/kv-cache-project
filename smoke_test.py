"""Offline check of paired CLI paths using an untrained local Qwen2 fixture."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer, decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from compare import compare, load
from missing_answer import SIX_DIGITS, classify
from needle import exact_match

HERE = Path(__file__).parent


def common_prefix(a, b):
    count = 0
    while count < min(len(a), len(b)) and a[count] == b[count]:
        count += 1
    return count


def check_needle(root, checkpoint):
    for cache in ("fp32", "int8"):
        subprocess.run([
            sys.executable, str(HERE / "needle.py"),
            "--model", str(checkpoint), "--revision", "local-smoke",
            "--cache", cache, "--lengths", "256", "--threads", "2",
            "--output", str(root / f"{cache}.jsonl"),
        ], check=True)
    fp32, int8 = (load(root / f"{cache}.jsonl") for cache in ("fp32", "int8"))
    summaries = list(compare(fp32, int8))
    assert len(summaries) == 3
    assert exact_match(" 123456.\n", "123456")
    assert not exact_match("The code is 123456", "123456")
    assert [row["needle_start_token"] for row in int8.values()] == sorted(
        row["needle_start_token"] for row in int8.values()
    )
    for key, before in fp32.items():
        after = int8[key]
        assert "task" not in before
        assert before["prefill_kv_bytes"] == 131072
        assert after["prefill_kv_bytes"] == 32768
        assert after["prefill_scale_bytes"] == 8192
    first = next(iter(fp32))
    int8[first]["prompt_sha256"] = "mismatch"
    try:
        list(compare(fp32, int8))
    except ValueError as exc:
        assert "prompt_sha256" in str(exc)
    else:
        raise AssertionError("Mismatched prompts were accepted")


def check_needle_options(root, checkpoint):
    """--depths and --task: custom positions are used and the label is written."""
    output = root / "early.jsonl"
    subprocess.run([
        sys.executable, str(HERE / "needle.py"),
        "--model", str(checkpoint), "--revision", "local-smoke", "--cache", "fp32",
        "--lengths", "256", "--depths", "0.01", "0.05", "--task", "needle-early",
        "--threads", "2", "--output", str(output),
    ], check=True)
    rows = list(load(output).values())
    assert sorted(row["depth"] for row in rows) == [0.01, 0.05]
    assert {row["task"] for row in rows} == {"needle-early"}


def check_missing_answer(root, checkpoint, length=256, trials=2):
    common = ["--model", str(checkpoint), "--revision", "local-smoke",
              "--lengths", str(length), "--trials", str(trials), "--threads", "2"]
    script = str(HERE / "missing_answer.py")

    # Prompts only: no model is loaded.
    subprocess.run([sys.executable, script, *common, "--dump-prompts", str(root / "prompts.jsonl")],
                   check=True)
    prompts = [json.loads(line) for line in (root / "prompts.jsonl").read_text().splitlines()]
    assert [p["variant"] for p in prompts] == ["absent", "present"] * trials
    for absent, present in zip(prompts[::2], prompts[1::2]):
        assert len(absent["token_ids"]) == len(present["token_ids"]) == length
        assert absent["depth"] is None and absent["expected"] == "NOT FOUND"
        assert not SIX_DIGITS.search(absent["text"])
        assert SIX_DIGITS.findall(present["text"]) == [present["expected"]]
        # Removing the needle from the present prompt must give the absent prompt minus one
        # contiguous run of extra filler tokens (absent = A + extra + C, without = A + C).
        start, end = present["needle_start_token"], present["needle_end_token_exclusive"]
        ids = present["token_ids"]
        without_needle = ids[:start] + ids[end:]
        assert common_prefix(absent["token_ids"], without_needle) + common_prefix(
            absent["token_ids"][::-1], without_needle[::-1]) >= len(without_needle)

    # Full runs on both CPU cache paths.
    for cache in ("fp32", "int8"):
        subprocess.run([sys.executable, script, *common, "--cache", cache,
                        "--output", str(root / f"missing-{cache}.jsonl")], check=True)
    fp32, int8 = (load(root / f"missing-{cache}.jsonl") for cache in ("fp32", "int8"))
    assert len(fp32) == 2 * trials
    for key, row in fp32.items():
        assert row["task"] == "missing-answer-v1"
        assert row["prompt_sha256"] == int8[key]["prompt_sha256"]
        assert row["correct"] == (row["outcome"] == ("abstain" if row["variant"] == "absent" else "target_code"))
    assert [s["depth"] for s in compare(fp32, int8)] == [None, 0.5]

    assert classify(" NOT FOUND.\n", None) == "abstain"
    assert classify("NOT FOUND", "123456") == "abstain"
    assert classify("not found", None) == "abstain"
    assert classify("Not Found.", None) == "abstain"
    assert classify("The code was not found", None) == "other"
    assert classify("123456.", "123456") == "target_code"
    assert classify("654321", "123456") == "other_code"
    assert classify("654321", None) == "other_code"
    assert classify("The code is 123456", "123456") == "other_code"
    assert classify("I cannot find it", None) == "other"
    return 2 * trials


def main():
    torch.manual_seed(226)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint"
        vocab = {letter: idx for idx, letter in enumerate(sorted(ByteLevel.alphabet()))}
        tokenizer = Tokenizer(BPE(vocab, []))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()  # decoded text must round-trip for the prompt checks
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            chat_template="{{ messages[0]['content'] }}\nAssistant:",
        )
        fast.save_pretrained(checkpoint)
        config = Qwen2Config(
            vocab_size=len(vocab), hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=512, eos_token_id=None, bos_token_id=None,
        )
        Qwen2ForCausalLM(config).save_pretrained(checkpoint)
        check_needle(root, checkpoint)
        check_needle_options(root, checkpoint)
        missing_cases = check_missing_answer(root, checkpoint)
    print(json.dumps({"status": "passed", "offline_paired_cases": 3,
                      "fp32_bytes": 131072, "int8_bytes_with_scales": 40960,
                      "missing_answer_paired_cases": missing_cases}))


if __name__ == "__main__":
    main()
