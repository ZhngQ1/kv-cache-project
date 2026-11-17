# Extended Qwen A100 retrieval results

## Validation and results

The extension contains 120 records per cache condition, covering prompt lengths 16,384 and 32,752, depths 10%, 50%, and 90%, and 20 trials per cell. Both conditions achieve 120/120 exact matches. Independent validation reproduced the uploaded comparison, recomputed exact-match scores, checked complete case coverage and paired input hashes, and verified that recorded needle positions fall inside each prompt.

Combining the extension with the earlier 512–8,192-token grid yields 420/420 exact matches per condition across seven lengths. Every one of the 21 length/depth cells scores 20/20 in both conditions. There are zero lost cases and zero gained cases under FP8.

| Prompt tokens | BF16, all three depths | FP8, all three depths |
| ---: | ---: | ---: |
| 512 | 60/60 | 60/60 |
| 1,024 | 60/60 | 60/60 |
| 2,048 | 60/60 | 60/60 |
| 4,096 | 60/60 | 60/60 |
| 8,192 | 60/60 | 60/60 |
| 16,384 | 60/60 | 60/60 |
| 32,752 | 60/60 | 60/60 |

## Reproducibility

The extension retains Qwen/Qwen2.5-0.5B-Instruct revision 7ae557604adf67be50417f59c2c2f167def9a775, NVIDIA A100-SXM4-40GB, BF16 weights, vLLM 0.10.1, PyTorch 2.7.1+cu126, V0, XFormers, greedy generation, and seed 2026. Startup logs confirm the same model and tokenizer revisions in both conditions. The native KV dtype is auto, following BF16 weights. FP8 uses fp8_e4m3 and dynamic scales. Prefix caching is disabled and each request is run individually.

Both extension engines use a maximum total sequence length of 32,768, which accommodates the longest prompt of 32,752 plus the maximum output budget of 16. The earlier grid used a maximum engine sequence length of 8,208. This changes profiling and pool allocation between the two batches. Within each batch, the two cache conditions share the same engine sequence limit.

## Memory observations

Extension logs reserve approximately 33.41 GiB for the KV pool in both conditions. The native pool contains 182,444 GPU blocks, while FP8 contains 364,889, approximately twice the capacity at the same block size. The earlier grid reserved 34.24 GiB for KV cache. Extension profiling reports a larger activation peak because its engine sequence limit is longer.

These engine logs measure allocation capacity at a fixed pool budget. Per-request allocated KV payload, retrieval latency, decode throughput, and lower total process GPU allocation were not measured by this script.

## Effective context and interpretation

Under the project's operational definition of at least 80% exact match at every tested depth and all shorter tested lengths, both caches support at least 32,752 prompt tokens on this task. This reaches the checkpoint's configured 32,768-token total context budget when the output allowance is included. The experiment therefore provides a lower bound on the effective context window for this particular retrieval setup. It does not locate a quality failure boundary.

The grid has a ceiling effect. The prompt contains one salient target code among synthetic background sentences. The same 20 trial seeds and codes are reused across lengths and depths, so the combined 420 cases share structure. Aggregate counts should not be analyzed as 420 independent trials. The evidence supports preserved single-needle retrieval under the tested configuration. Natural-document, plausible-distractor, and multi-needle tasks would provide additional coverage.

The planned length sweep is complete. Any harder retrieval task should be labeled as a separate follow-up, with both cache conditions evaluated under the same revised prompt. Longer-than-configured contexts would require additional model or position-encoding assumptions.

## Files

The original full-*.jsonl and full-*.log files contain this extension. settings.json and comparison.jsonl are copied from the uploaded ZIP. combined-auto.jsonl and combined-fp8_e4m3.jsonl merge the earlier grid with the extension using distinct case keys. combined-accuracy-summary.csv and combined-retrieval-heatmaps.png/.svg summarize all seven lengths.
