# A100 FP16 baseline results

This directory contains the uploaded Colab measurements and derived figures. All 36 records passed independent checks of the dense KV payload formula, the six expected lengths, and repeat IDs 0–2 for each architecture and length.

The run used NVIDIA A100-SXM4-40GB, PyTorch 2.14.1+cu130, Transformers 4.57.1, and SDPA. Both randomly initialized Llama configurations use four layers, eight query heads, head dimension 32, batch size 1, and FP16. KV heads are eight for MHA and two for GQA. K/V projection parameter counts also change.

| Metric at 8,192 tokens | MHA | GQA |
| --- | ---: | ---: |
| Measured KV payload (GB) | 0.033554432 | 0.008388608 |
| Peak CUDA allocated memory (GB) | 0.090706944 | 0.064754688 |
| Peak increase above pre-forward allocation (GB) | 0.076611584 | 0.051445760 |
| Mean prefill latency (ms) | 5.309 | 5.120 |
| Sample standard deviation (ms), n = 3 | 0.020 | 0.030 |

KV payload grows linearly at 4,096 bytes per token for MHA and 1,024 bytes per token for GQA. Reducing KV heads from eight to two reduces measured KV payload by 75%. Absolute peak allocation falls by approximately 28.6% at 8,192 tokens. It includes model allocations and temporary tensors. Prefill time is approximately 3.6% lower for GQA in this small run. Three repeats and a small model provide preliminary latency observations.

The SDPA setting records the PyTorch attention interface. The specific CUDA attention kernel selected internally was not captured. The earlier CPU run used FP32 and a different PyTorch version. Cross-run differences therefore include precision, runtime, and hardware effects.

The benchmark evaluates memory scaling with random weights. Pretrained quality, decode throughput, FP8, larger models, and additional independent timing runs remain to be measured.

Files include raw baseline.jsonl, summary.csv, and PNG/SVG versions of the cache-growth, peak-memory, and prefill-latency figures. GB uses decimal units. Error bars in the latency figure show sample standard deviation across three measured repeats.
