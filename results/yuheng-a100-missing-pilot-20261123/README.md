# Missing-answer pilot

This pilot checked the prompt wording and scoring of the missing-answer control before the main run in [yuheng-a100-missing-20261123](../yuheng-a100-missing-20261123/README.md). It is a protocol check and is not part of the main results.

## Task

Each trial produces two 512-token prompts that share the same synthetic archive and the same instruction:

- **absent**: the archive contains no verification code. The correct output is NOT FOUND.
- **present**: the same archive with the code inserted at 50% depth. The correct output is the code.

The archive, needle sentence, and chat template are the same as in the single-needle task. Only the question differs:

> Question: What is the archive's verification code? If the archive does not contain a verification code, reply NOT FOUND. Reply with only the code or NOT FOUND.

Answers are classified as abstain (NOT FOUND in any letter case, optional final period), target_code (exactly the code), other_code (another six-digit number), or other. The run used five trials with seed 226, task version missing-answer-v1.

## Results

| Trial | Absent, BF16 and FP8 | Present, BF16 and FP8 | Expected code |
| ---: | --- | --- | --- |
| 0 | NOT FOUND | NOT FOUND | 607564 |
| 1 | NOT FOUND | 518442 | 518442 |
| 2 | NOT FOUND | NOT FOUND | 892183 |
| 3 | NOT FOUND | 654493 | 654493 |
| 4 | NOT FOUND | 839241 | 839241 |

Both cache conditions produced identical answers on all ten prompts. All absent prompts were answered NOT FOUND. Two of five present prompts were also answered NOT FOUND, although the code was in the prompt. No answer contained an invented code or extra text.

## Decision

The wording was frozen unchanged for the main run. The experiment compares cache precision, and the two conditions behaved identically. The tendency to abstain when the code is present is reported as a result rather than tuned away on five prompts. The present control exists to expose this kind of bias.

The main run uses the same seed, so its 512-token trials 0–4 are the same prompts as this pilot. Their answers in the main run are identical to the answers above.

## Environment

The run used Modal on an NVIDIA A100-SXM4-40GB with the reference software configuration (Python 3.12.16, vLLM 0.10.1, PyTorch 2.7.1, Transformers 4.57.1, V0 engine, XFormers backend, BF16 weights). Evaluation code is commit a7ebc90. The reproduction in [yuheng-a100-repro-20261123](../yuheng-a100-repro-20261123/README.md) verifies this environment against the original pilot. The engine sequence limit is 528 in both conditions. Logged GPU cache blocks are 188,516 for BF16 and 377,033 for FP8 within a 34.52 GiB pool.

## Files

- missing-prompts.jsonl contains the token IDs and decoded text of all ten prompts.
- missing-auto.jsonl and missing-fp8_e4m3.jsonl contain the per-case records, including variant and outcome.
- missing-auto.log and missing-fp8_e4m3.log contain the complete startup logs and outcome counts.
- comparison.jsonl is the paired summary from compare.py. The row with depth null is the absent variant.
- settings.json, nvidia-smi.txt, python-version.txt, pip-freeze.txt, and code-commit.txt record the configuration and environment.
