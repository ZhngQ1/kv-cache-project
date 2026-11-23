# Missing-answer control

## Task

This control tests whether the model reports that an answer is absent instead of producing a code. Each trial produces two prompts of the same length that share the same synthetic archive and the same instruction:

- **absent**: the archive contains no verification code. The correct output is NOT FOUND.
- **present**: the same archive with the code inserted at 50% depth. The correct output is the code.

The present prompts check whether the instruction makes the model abstain when the answer is available. Without them, a model that always answers NOT FOUND would score perfectly. The archive, needle sentence, and chat template match the single-needle task. The question, frozen after the [pilot](../yuheng-a100-missing-pilot-20261123/README.md), is:

> Question: What is the archive's verification code? If the archive does not contain a verification code, reply NOT FOUND. Reply with only the code or NOT FOUND.

Each answer is classified as abstain (NOT FOUND in any letter case, optional final period), target_code (exactly the code), other_code (another six-digit number), or other. Absent cases are correct only for abstain. Present cases are correct only for target_code. Abstention and retrieval are reported separately. The task version is missing-answer-v1.

## Results

| Prompt tokens | Variant | BF16 | FP8 |
| ---: | --- | --- | --- |
| 512 | absent | abstain 10/10 | abstain 10/10 |
| 512 | present | target_code 5/10, abstain 5/10 | target_code 5/10, abstain 5/10 |
| 4,096 | absent | abstain 10/10 | abstain 10/10 |
| 4,096 | present | target_code 10/10 | target_code 10/10 |

The two cache conditions produced identical answers on all 40 prompts. Every absent prompt was answered NOT FOUND. No answer, in either variant, contained an invented or incorrect code or any text other than NOT FOUND or a code. At 512 tokens, five present prompts (trials 0, 2, 5, 6, and 9) were answered NOT FOUND although the code was in the prompt. At 4,096 tokens, every present prompt was answered correctly.

## Interpretation

FP8 KV storage did not change abstention or retrieval in this control.

The present prompts show that the instruction biases this model toward NOT FOUND at 512 tokens. The perfect abstention on absent prompts therefore partly reflects that bias and should not be read as reliable detection of missing information. The single-needle task without the NOT FOUND instruction scored 100% at 512 tokens in the primary grid, so the false abstentions follow from the added instruction rather than from cache precision. False abstention occurred at 512 but not at 4,096 tokens. With ten trials per cell, this run does not establish why.

## Experimental setup

The model is Qwen/Qwen2.5-0.5B-Instruct at revision 7ae557604adf67be50417f59c2c2f167def9a775. Startup logs confirm the model and tokenizer revisions, BF16 weights, the V0 engine, and the XFormers backend in both conditions. The run used Modal on an NVIDIA A100-SXM4-40GB with Python 3.12.16, vLLM 0.10.1, PyTorch 2.7.1, and Transformers 4.57.1. The native cache follows the BF16 model dtype. The quantized condition uses fp8_e4m3 with calculate_kv_scales=True. Both use eager execution, one concurrent sequence, disabled prefix caching, greedy decoding, a maximum of 16 generated tokens, and a maximum total sequence length of 4,112. Both conditions ran sequentially in one container on one GPU with identical prompts and request order. Evaluation code is commit a7ebc90 (`missing_answer.py`).

The run uses seed 226 with ten trials at 512 and 4,096 tokens, giving 40 prompts per condition. Each present prompt is its absent prompt with the needle inserted and an equal number of filler tokens removed from the end of the archive. Generation checks that absent prompts contain no six-digit number and that present prompts contain only the target code.

## Cache capacity

| Logged quantity | BF16 cache | FP8 cache |
| --- | ---: | ---: |
| KV pool budget (GiB) | 34.40 | 34.40 |
| GPU cache blocks | 187,853 | 375,707 |

## Limits

Each cell has ten trials. The 512-token trials 0–4 are the same prompts as the pilot, because both runs use seed 226, and their answers are identical to the pilot. The task uses synthetic archives with a single possible code. Distractor codes and natural documents were not tested.

## Files

- missing-prompts.jsonl contains the token IDs and decoded text of all 40 prompts.
- missing-auto.jsonl and missing-fp8_e4m3.jsonl contain the per-case records, including variant, outcome, and prompt hashes.
- missing-auto.log and missing-fp8_e4m3.log contain the complete startup logs and outcome counts.
- comparison.jsonl is the paired summary from compare.py. Rows with depth null are absent prompts; rows with depth 0.5 are present prompts.
- settings.json, nvidia-smi.txt, python-version.txt, pip-freeze.txt, and code-commit.txt record the configuration and environment.
