# A100 FP16 results with 20 repeats

This run is the updated timing experiment. The raw file contains 240 measured forward passes across six lengths, two architectures, and 20 repeats per condition. All records pass independent checks of the dense KV formula and the expected repeat IDs. The three-repeat pilot remains in ../charles-a100-fp16/.

## Configuration

The run used NVIDIA A100-SXM4-40GB, PyTorch 2.14.1+cu130, Transformers 4.57.1, and SDPA. Both random Llama models have four layers, hidden dimension 256, eight query heads, head dimension 32, batch size one, and FP16. MHA has eight KV heads and GQA has two. K/V projection sizes and total parameter counts also change, from 2,754,816 to 2,361,600 parameters. Each condition has one warmup followed by 20 timed forward passes. Timing uses host wall time with CUDA synchronization and covers the model body, excluding the LM head, tokenization, and generation.

## Memory results

| Metric at 8,192 tokens | MHA | GQA |
| --- | ---: | ---: |
| Measured KV payload (GB) | 0.033554432 | 0.008388608 |
| Peak CUDA allocated memory (GB) | 0.090706944 | 0.064754688 |
| Peak increase above pre-forward allocation (GB) | 0.076611584 | 0.051445760 |

KV payload grows at 4,096 bytes per token for MHA and 1,024 bytes per token for GQA. Reducing KV heads from eight to two reduces payload by 75% at every measured length. Absolute peak allocation is approximately 28.6% lower at 8,192 tokens. It includes model allocations and temporary tensors. These memory measurements agree with the earlier three-repeat pilot. GB uses decimal units.

## Prefill timing

Values below are mean ± sample standard deviation in milliseconds, with 20 observations per condition. The final column is 100 × (MHA mean − GQA mean) / MHA mean. Positive values indicate lower GQA latency. Error bars show sample variability across repeats.

| Prompt tokens | MHA (ms) | GQA (ms) | GQA latency reduction |
| ---: | ---: | ---: | ---: |
| 128 | 5.116 ± 0.098 | 5.107 ± 0.021 | +0.16% |
| 512 | 5.095 ± 0.032 | 5.075 ± 0.084 | +0.41% |
| 1,024 | 4.862 ± 0.019 | 4.935 ± 0.086 | -1.50% |
| 2,048 | 5.216 ± 0.123 | 5.202 ± 0.150 | +0.27% |
| 4,096 | 5.208 ± 0.031 | 5.165 ± 0.040 | +0.82% |
| 8,192 | 5.341 ± 0.023 | 4.948 ± 0.023 | +7.36% |

At 8,192 tokens, GQA averages 4.948 ms versus 5.341 ms for MHA, a 7.36% reduction. Median times are 4.944 ms and 5.343 ms. The earlier pilot showed a 3.6% mean reduction at that length. Across lengths the timing advantage varies, and GQA is slower at 1,024 tokens.

All repeats were collected sequentially in one runtime session, with MHA conditions preceding GQA conditions. They measure variability within that execution. Run order, GPU clock state, and host overhead may influence the observed differences. The model is small and total prefill time stays near five milliseconds. Broad speedup claims would require independent sessions, more warmup, interleaved run order, and larger workloads.

## Interpretation and limits

The memory measurements support linear KV growth and a fourfold reduction in stored KV payload for this controlled configuration. The latency data provide preliminary observations for this workload. SDPA records the PyTorch interface. The internally selected CUDA attention kernel was not captured. CPU FP32 and GPU FP16 runs also differ in runtime version and hardware.

The model weights are random. Pretrained retrieval quality, FP8 cache accuracy, and decode throughput are evaluated in the team's other experiments.

## Files

- baseline.jsonl contains all 240 original uploaded records.
- summary.csv contains per-condition memory, mean latency, median latency, and sample standard deviation.
- kv-cache-growth.png and .svg show measured KV payload.
- peak-memory.png and .svg show absolute peak CUDA allocation.
- prefill-latency.png and .svg show mean prefill time and sample standard deviation.
