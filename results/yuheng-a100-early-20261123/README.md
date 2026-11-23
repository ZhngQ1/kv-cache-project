# Early-position retrieval

## Results

Both native BF16 and FP8 E4M3 KV caches achieve 60/60 exact matches with the needle placed at 1% and 5% depth. The grid covers three prompt lengths, two depths, and ten trials per cell. There are zero lost and zero gained cases under FP8, and the two conditions produced identical answers on all 60 prompts.

| Prompt tokens | BF16 at 1% / 5% | FP8 at 1% / 5% |
| ---: | --- | --- |
| 4,096 | 10/10 · 10/10 | 10/10 · 10/10 |
| 16,384 | 10/10 · 10/10 | 10/10 · 10/10 |
| 32,752 | 10/10 · 10/10 | 10/10 · 10/10 |

Depth is the needle's insertion point as a fraction of filler tokens, as in the primary grid. The absolute needle positions and the number of tokens between the needle and the end of the prompt are fixed per cell:

| Prompt tokens | Depth | Needle starts at token | Tokens after the needle |
| ---: | ---: | ---: | ---: |
| 4,096 | 1% | 69 | 4,012 |
| 4,096 | 5% | 231 | 3,850 |
| 16,384 | 1% | 192 | 16,177 |
| 16,384 | 5% | 845 | 15,524 |
| 32,752 | 1% | 356 | 32,381 |
| 32,752 | 5% | 1,663 | 31,074 |

## Relation to the primary grid

The task is the existing single-needle task with unchanged prompt text and scoring. The run uses seed 2026 and trials 0–9, the same seed as the primary grid. These ten trials therefore have the same codes and the same archive text as trials 0–9 of [qwen-a100-full-20261117](../qwen-a100-full-20261117/README.md) and [qwen-a100-long-20261117](../qwen-a100-long-20261117/README.md). Only the needle position differs. In those runs the same trials score 90/90 at 10%, 50%, and 90% depth for these three lengths in both conditions. Across all five tested depths, these trials are retrieved correctly at every length. The earlier depths ran on Colab and these on Modal. The reproduction in [yuheng-a100-repro-20261123](../yuheng-a100-repro-20261123/README.md) shows identical prompts and answers across the two environments.

## Effective context

Under the project's operational definition, at least 80% exact match at every tested depth and every shorter tested length, both caches support at least 32,752 prompt tokens on this task, now including needles in the first 1% of the filler. In the longest cell the needle starts at token 356 and is followed by more than 32,000 tokens before the question. The experiment does not locate a failure boundary, so the effective context window remains a lower bound.

## Experimental setup

The model is Qwen/Qwen2.5-0.5B-Instruct at revision 7ae557604adf67be50417f59c2c2f167def9a775. Startup logs confirm the model and tokenizer revisions, BF16 weights, the V0 engine, and the XFormers backend in both conditions. The run used Modal on an NVIDIA A100-SXM4-40GB with Python 3.12.16, vLLM 0.10.1, PyTorch 2.7.1, and Transformers 4.57.1. The native cache follows the BF16 model dtype. The quantized condition uses fp8_e4m3 with calculate_kv_scales=True. Both use eager execution, one concurrent sequence, disabled prefix caching, greedy decoding, a maximum of 16 generated tokens, and a maximum total sequence length of 32,768. Both conditions ran sequentially in one container on one GPU with identical prompts and request order. Evaluation code is commit a7ebc90 (`needle.py --depths 0.01 0.05 --task needle-early`).

## Cache capacity

| Logged quantity | BF16 cache | FP8 cache |
| --- | ---: | ---: |
| KV pool budget (GiB) | 33.43 | 33.43 |
| GPU cache blocks | 182,551 | 365,103 |

These figures are close to the long-context extension, which used the same 32,768-token engine limit and logged 182,444 and 364,889 blocks in a 33.41 GiB pool. FP8 provides twice the block count at the same budget. The script does not measure latency, throughput, or per-request payload.

## Interpretation and limits

Moving the needle to the start of the prompt did not reduce retrieval in either cache condition. FP8 KV storage with dynamic scales again preserves exact-match performance.

The grid has a ceiling effect. With ten trials per cell and no failures, a one-sided 95% exact binomial bound limits the per-cell failure rate to about 26%. Smaller degradations cannot be excluded. The six cells reuse the same ten codes and archives, so the 60 cases are not independent draws. The task contains one salient needle in synthetic background sentences. FP8 quantizes cached keys and values but does not evict tokens. Eviction methods such as StreamingLLM or H₂O could lose an early needle and were not tested.

## Files

- early-auto.jsonl and early-fp8_e4m3.jsonl contain the per-case records, including needle positions and prompt hashes.
- early-auto.log and early-fp8_e4m3.log contain the complete startup and completion logs.
- comparison.jsonl is the paired summary from compare.py.
- settings.json records the configuration, GPU model, and code commit.
- nvidia-smi.txt, python-version.txt, and pip-freeze.txt record the environment.
- code-commit.txt records the evaluation code commit.
