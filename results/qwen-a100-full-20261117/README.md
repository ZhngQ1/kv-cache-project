# Qwen A100 full retrieval evaluation

## Results

Both native BF16 and FP8 E4M3 KV caches achieve 300/300 exact matches. Each condition covers five lengths, three insertion depths, and 20 trials per cell. Independent checks reproduced the uploaded comparison, verified all 600 exact-match scores, and confirmed identical prompt hashes and evaluation metadata for each pair. Each condition contains 300 distinct prompt hashes.

| Prompt tokens | BF16 at 10% / 50% / 90% | FP8 at 10% / 50% / 90% |
| ---: | --- | --- |
| 512 | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 |
| 1,024 | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 |
| 2,048 | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 |
| 4,096 | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 |
| 8,192 | 20/20 · 20/20 · 20/20 | 20/20 · 20/20 · 20/20 |

The paired accuracy difference is zero percentage points in every cell. There are zero lost cases and zero gained cases. Using the project's 80% threshold across all depths and all shorter tested lengths, both conditions support an effective context window of at least 8,192 tokens on this task. The failure boundary remains unobserved.

## Experimental setup

The model is Qwen/Qwen2.5-0.5B-Instruct at revision 7ae557604adf67be50417f59c2c2f167def9a775. Startup logs confirm the same model and tokenizer revision in both runs. Hardware is NVIDIA A100-SXM4-40GB. Both conditions use BF16 model weights, vLLM 0.10.1, PyTorch 2.7.1+cu126, the V0 engine, and the XFormers backend.

The native condition uses kv_cache_dtype=auto, which follows the BF16 model dtype. The quantized condition uses fp8_e4m3 with calculate_kv_scales=True. Both use eager execution, a maximum total sequence length of 8,208, one concurrent sequence, disabled prefix caching, greedy decoding, and a maximum of 16 generated tokens.

The evaluation uses seed 2026, distinct from the pilot seed 226. Prompt length includes the chat template and question. A random six-digit code is inserted into a synthetic archive of background sentences. Depth specifies the insertion point as a fraction of filler tokens. The output must match the target code after trimming whitespace and allowing an optional final period. Absolute needle token positions and input hashes are retained in the raw records.

## Cache capacity

| Logged quantity | BF16 cache | FP8 cache |
| --- | ---: | ---: |
| KV pool budget, rounded (GiB) | 34.24 | 34.24 |
| GPU cache blocks | 186,988 | 373,977 |

With the same engine block size, the logged FP8 allocation supplies approximately twice the token capacity under the same pool budget. vLLM reserves a large KV pool in both conditions. These logs support a capacity improvement at a fixed budget. Per-request allocated payload and reduced total process GPU memory were not measured by this retrieval script. The engine's logged maximum concurrency is a capacity estimate, while this evaluation processes one request at a time.

## Interpretation

Across the tested 512–8,192-token range, FP8 E4M3 cache storage with dynamic scales preserves exact-match performance on this single-code retrieval task. The full score grid has a ceiling effect, which limits its ability to distinguish the two conditions. The result provides evidence of preserved retrieval for the tested inputs and configuration.

The same 20 trial seeds and target codes are reused across depths and lengths. These 300 cases therefore share structure, and should not be treated as 300 independent draws for aggregate uncertainty estimates. Each individual cell contains 20 trials. The task uses synthetic background sentences and one salient needle. Distractor-rich, multi-needle, and natural-document retrieval remain outside this evaluation.

The script does not record retrieval latency or decode throughput. Its results do not support speedup claims. Logs contain a process-group shutdown warning after successful completion in both conditions.

## Suggested follow-up

Retain this grid as the completed primary experiment. To investigate the effective-context boundary, extend the frozen protocol to longer prompts within the checkpoint's configured context limit. If the extended grid also saturates, report a lower bound and consider a separately labeled task with plausible distractor codes. Any changed prompt should be evaluated in both cache conditions.

## Files

- full-auto.jsonl and full-fp8_e4m3.jsonl contain the original per-case records.
- full-auto.log and full-fp8_e4m3.log contain the complete startup and completion logs.
- settings.json records the submitted configuration.
- comparison.jsonl is the uploaded paired summary, independently reproduced during analysis.
- accuracy-summary.csv provides the same per-cell comparison for plotting and tables.
- cache-capacity.csv records cache pool budgets and block counts from the logs.
- retrieval-heatmaps.png and retrieval-heatmaps.svg show BF16 accuracy, FP8 accuracy, and their paired difference.
