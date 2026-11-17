# Charles's CPU baseline

Measured on November 17, 2026 using ARM64 macOS, PyTorch 2.9.1 and Transformers 4.57.1.

The sweep contains 36 forward passes across six prompt lengths, two KV-head settings, and three measured repeats after one warmup per configuration. Both randomly initialized Llama configurations use four layers, eight query heads, head dimension 32, batch size 1, FP32 and SDPA.

Every measured cache payload equals the dense-cache formula. MHA uses 8,192 bytes per token and GQA uses 2,048 bytes per token. At 8,192 tokens the measured payloads are 0.067108864 GB and 0.016777216 GB, respectively. GQA reduces KV payload by 75% in this configuration.

K/V projection parameter counts change with KV-head count. The experiment measures architecture-level cache growth with random weights. Pretrained retrieval quality and GPU memory require their own measurements. Prefill time is an auxiliary CPU measurement and varies with system load.

Files include raw JSONL, a CSV summary with mean and sample standard deviation for prefill time, and PNG/SVG figures. The repeated payload measurements coincide exactly.

To reproduce from the repository root in an environment with the base dependencies installed, run the following commands. Use a fresh result directory to retain earlier measurements.

```bash
python -m pip install -r requirements-plots.txt
mkdir -p results/cpu-reproduction
python baseline.py --lengths 128 512 1024 2048 4096 8192 > results/cpu-reproduction/baseline.jsonl
python plot_baseline.py results/cpu-reproduction
```
