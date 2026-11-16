"""Offline check of paired CLI paths using an untrained local Qwen2 fixture."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from compare import compare, load
from needle import exact_match


def main():
    torch.manual_seed(226)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "checkpoint"
        vocab = {letter: idx for idx, letter in enumerate(sorted(ByteLevel.alphabet()))}
        tokenizer = Tokenizer(BPE(vocab, []))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
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
        for cache in ("fp32", "int8"):
            subprocess.run([
                sys.executable, str(Path(__file__).with_name("needle.py")),
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
    print(json.dumps({"status": "passed", "offline_paired_cases": 3,
                      "fp32_bytes": 131072, "int8_bytes_with_scales": 40960}))


if __name__ == "__main__":
    main()
