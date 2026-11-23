# Independent reproduction of the Qwen A100 pilot

This run repeats the 512-token pilot in [qwen-a100-pilot-20261117](../qwen-a100-pilot-20261117/README.md) on a separate platform before the follow-up experiments. It verifies that the environment used for the early-position and missing-answer runs matches the reference configuration. It is not part of the follow-up results.

## Results

Both cache conditions achieve 15/15 exact matches. Each of three depths (10%, 50%, 90%) contains five paired trials with seed 226, as in the original pilot.

| Depth | Native BF16 hits | FP8 hits |
| --- | ---: | ---: |
| 10% | 5/5 | 5/5 |
| 50% | 5/5 | 5/5 |
| 90% | 5/5 | 5/5 |

Compared case by case with the original pilot, all 15 prompt hashes match in each condition, and all 15 generated answers are identical in each condition. The paired BF16/FP8 difference is zero, with no lost or gained cases.

## Environment

The run used Modal rather than Colab, on an NVIDIA A100-SXM4-40GB (driver 610.57.04). The software matches the reference configuration: Python 3.12.16, vLLM 0.10.1, PyTorch 2.7.1, Transformers 4.57.1, xformers 0.0.31, the V0 engine, and the XFormers backend. Startup logs confirm Qwen/Qwen2.5-0.5B-Instruct at model and tokenizer revision 7ae557604adf67be50417f59c2c2f167def9a775, BF16 weights, eager execution, disabled prefix caching, and a maximum sequence length of 528. The native cache follows the BF16 model dtype. The FP8 condition uses fp8_e4m3 with calculate_kv_scales=True. Evaluation code is commit a7ebc90 on the yuheng-dev branch. Both conditions ran sequentially in one container on one GPU.

## Cache capacity

| Logged quantity | This run, BF16 | This run, FP8 | Original pilot, BF16 | Original pilot, FP8 |
| --- | ---: | ---: | ---: | ---: |
| KV pool budget (GiB) | 34.52 | 34.52 | 34.50 | 34.50 |
| GPU cache blocks | 188,539 | 377,079 | 188,409 | 376,819 |

The KV pool on this machine is about 0.07% larger than in the original run, which is consistent with small differences in driver and runtime memory overhead. FP8 again provides twice the block count of BF16 at the same budget.

## Interpretation

Identical prompts, identical answers, and nearly identical cache capacity indicate that this environment reproduces the reference configuration for retrieval accuracy. The early-position and missing-answer runs used the same Modal environment. Logs end with a process-group shutdown warning after successful scoring, as in the original runs.

## Files

- repro-auto.jsonl and repro-fp8_e4m3.jsonl contain the per-case records.
- repro-auto.log and repro-fp8_e4m3.log contain the complete startup and completion logs.
- comparison.jsonl is the paired summary from compare.py.
- settings.json records the configuration, GPU model, and code commit.
- nvidia-smi.txt, python-version.txt, and pip-freeze.txt record the environment.
- code-commit.txt records the evaluation code commit.
