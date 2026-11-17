# Qwen A100 retrieval pilot

The 512-token pilot passed all 15 cases in both conditions. Each of three depths (10%, 50%, 90%) contains five paired trials. Independent checks reproduced the saved comparison and verified exact-match scores and paired input hashes.

The run uses Qwen2.5-0.5B-Instruct revision 7ae557604adf67be50417f59c2c2f167def9a775, NVIDIA A100-SXM4-40GB, vLLM 0.10.1, PyTorch 2.7.1+cu126, BF16 model weights, V0 engine, XFormers, eager execution, and disabled prefix caching. Startup logs confirm both model and tokenizer revisions. The native cache follows BF16 weights. The FP8 condition uses fp8_e4m3 with calculate_kv_scales=True.

| Depth | Native BF16 hits | FP8 hits |
| --- | ---: | ---: |
| 10% | 5/5 | 5/5 |
| 50% | 5/5 | 5/5 |
| 90% | 5/5 | 5/5 |

Logs show approximately 34.50 GiB reserved for KV cache in both conditions. Native and FP8 have 188,409 and 376,819 GPU blocks, respectively, at the same default block size. The FP8 run therefore provides approximately twice the cache token capacity under the same pool budget. These are engine capacity figures. Only one request runs at a time, and the logged maximum concurrency is a capacity estimate.

The data establish that both inference paths work and that this short pilot shows no observed accuracy difference. Five trials reused across three positions provide limited coverage. Long-context retrieval and the effective context window require the full sweep. Logs end with a process-group shutdown warning after successful scoring.

Proceed with the frozen model and prompt, lengths 512/1024/2048/4096/8192, 20 trials per depth, and a fresh seed of 2026. This yields 300 cases per cache condition.
