# KV Cache Optimization

This course project measures how grouped-query attention (GQA) changes KV cache growth and how cache quantization affects retrieval from long prompts. The project runs on CPU, with a vLLM FP8 experiment when a compatible GPU is available.

## Team responsibilities

| Owner | Workstream | First deliverable |
| --- | --- | --- |
| Charles | MHA/GQA benchmark and project integration | Reproduce the CPU baseline and plot measured KV cache size in GB against prompt length. If Colab provides a GPU, run the CUDA baseline and save the GPU details and raw results. |
| Teammate B | GPU setup and FP8 experiment | Run a 512-token vLLM pilot with native and FP8 KV caches on the same GPU. Share both JSONL files and startup logs showing the cache dtype and attention backend. Try Colab first. Use a single on-demand L4 or RTX 4090 on Vast.ai if a compatible Colab session is unavailable. |
| Teammate C | Needle-in-a-Haystack evaluation and analysis | Run the 512-token CPU FP32/INT8 pilot, review the outputs by hand, and share a table by needle depth with a few failure examples. Own the final accuracy heatmaps and effective-context analysis. |

At the first check-in, review the unquantized pilot together and choose the model for the full evaluation. Each owner drafts the methods and results for their experiment. Teammate C starts the related-work outline, and all three review the figures, claims, and final paper.

| Date | Target |
| --- | --- |
| Nov 16–18 | Reproduce the baseline, complete the CPU retrieval pilot, and confirm GPU availability. |
| Nov 19–24 | Freeze the model and evaluation settings. Run the CPU retrieval grid and start the FP8 grid if the GPU pilot passes. |
| Nov 25–30 | Finish paired runs, verify the raw data, and finalize the memory curve and retrieval heatmaps. |
| Dec 3 | Course presentation. |
| Dec 10 | Submit the final paper. |

## Quick start

From the repository root, create an isolated environment with Python 3.12 or 3.13. The offline checks were verified on ARM64 macOS with Python 3.13.5, PyTorch 2.9.1, and Transformers 4.57.1.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p runs

python baseline.py --lengths 128 512 1024 2048 4096 8192 > runs/mha-gqa-cpu.jsonl
python smoke_test.py
```

- `baseline.py` measures cache bytes and prefill time for MHA and GQA.
- `needle.py` runs token-matched retrieval cases and records per-case results.
- `int8_cache.py` implements CPU INT8 cache storage with FP32 scales and dequantization for attention.
- `compare.py` checks paired inputs and summarizes accuracy by prompt length and needle depth.
- `smoke_test.py` checks the CPU retrieval pipeline offline with a random local Qwen2 fixture.
- `cpu-baseline.jsonl` contains the 36 measured runs. The `runs/` directory is ignored by Git. Share new raw results and logs with the team after each run.

## 1. MHA vs. GQA

The controlled benchmark compares two randomly initialized Llama models. Both use four layers, eight query heads, head dimension 32, FP32, batch size 1, identical input IDs, and the same attention backend. The changed configuration is the number of KV heads, which also changes the size of the K/V projections. The script warms up each length once, then measures three repeats.

The dense KV payload in bytes is `2 × batch × layers × tokens × KV heads × head dimension × bytes per element`. The script sums the actual K/V tensor bytes and checks them against this formula on every run. Use decimal GB (1 GB = 10⁹ bytes).

| Prompt tokens | MHA, 8 KV heads (GB) | GQA, 2 KV heads (GB) |
| ---: | ---: | ---: |
| 1,024 | 0.008389 | 0.002097 |
| 4,096 | 0.033554 | 0.008389 |
| 8,192 | 0.067109 | 0.016777 |

Measured slopes were 8,192 and 2,048 bytes per token, respectively. The 75% reduction applies to the KV tensor payload. Report allocator peak memory and prefill time as separate metrics. The prefill timer covers the model body forward pass and excludes tokenization, the LM head, and generation.

On a compatible CUDA runtime with the baseline dependencies installed, run:

```bash
nvidia-smi
python baseline.py --device cuda --dtype float16 \
  --lengths 128 512 1024 2048 4096 8192 > runs/mha-gqa-cuda.jsonl
```

Keep CPU FP32 and CUDA FP16 results labeled by device and dtype. On CUDA the script records both absolute peak allocation and the increase over the allocation immediately before the forward pass. The absolute peak includes model weights and temporary forward tensors.

## 2. CPU retrieval pilot

Use [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) from a machine that can reach Hugging Face, or supply a local checkpoint with `--model`. Use the same checkpoint and tokenizer in both conditions. Replace the value below with the full Hub commit SHA. Use new output names when repeating a run.

```bash
MODEL_REV=REPLACE_WITH_MODEL_COMMIT_SHA
for cache in fp32 int8; do
  python needle.py --revision "$MODEL_REV" --cache "$cache" \
    --lengths 512 --trials 5 --output "runs/pilot-${cache}.jsonl" || break
done
python compare.py runs/pilot-fp32.jsonl runs/pilot-int8.jsonl
```

Review the 15 cases before expanding the grid. If the FP32 baseline scores below 12/15, inspect the prompts, generated answers, and scoring. If the small model still struggles, repeat both conditions with [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) and its matching revision.

For the full experiment, use prompt lengths of 512, 1,024, 2,048, 4,096, and 8,192 tokens, with 20 paired trials at each of three needle depths (10%, 50%, and 90%). Use a fresh seed for the full run, such as `--seed 2026`. If time is tight, finish the paired runs at 512, 2,048, and 4,096 tokens first. Length includes the chat template and question. Depth is the needle's insertion point as a fraction of the filler tokens. Generation is greedy with up to 16 new tokens. An answer counts as correct if it contains only the target code after trimming surrounding whitespace and an optional final period.

Report exact-match counts, percentage-point differences, and the number of paired cases lost or gained by quantization. As a working definition of effective context window, report the longest tested length for which all three depths achieve at least 80% exact match at that length and every shorter tested length. Also show accuracy by depth, especially near the beginning of the prompt.

The CPU INT8 implementation includes scale bytes in the stored-cache measurement. It dequantizes before attention and creates temporary tensors. Treat the payload measurement and retrieval accuracy as the reported outcomes. The offline smoke test verifies both code paths with random weights. Pretrained retrieval accuracy is pending.

## 3. GPU FP8 pilot

Use a separate Linux GPU environment with a compatible vLLM build. The script targets vLLM 0.10.1. On CUDA, install `requirements-vllm.txt` in that environment. For ROCm, use a compatible build or container. Check the GPU and selected attention backend before committing to the full sweep. [vLLM FP8 documentation](https://docs.vllm.ai/en/v0.10.1/features/quantization/quantized_kvcache.html)

```bash
nvidia-smi  # NVIDIA environments
mkdir -p runs
MODEL_REV=REPLACE_WITH_MODEL_COMMIT_SHA
for cache in auto fp8_e4m3; do
  python needle.py --revision "$MODEL_REV" --cache "$cache" --gpu-dtype bfloat16 \
    --lengths 512 --trials 5 --output "runs/pilot-${cache}.jsonl" \
    > "runs/vllm-${cache}.log" 2>&1 || break
done
python compare.py runs/pilot-auto.jsonl runs/pilot-fp8_e4m3.jsonl
```

Use `--gpu-dtype float16` for both conditions if the GPU requires it. Keep both runs on the same GPU, with identical model weights, prompts, vLLM version, and generation settings. Record the resolved cache dtype, scale policy, attention backend, and KV capacity from the logs. If the backend changes with the cache setting, report the result as a comparison of two vLLM configurations. Distinguish theoretical payload and actual cache capacity from total GPU allocation, which can include a preallocated cache pool. GPU FP8 has yet to be verified.

## Related work

Cover cache compression, sparse access and eviction, and architectural simplification. Discuss PagedAttention under cache allocation and sharing. [GQA](https://arxiv.org/abs/2305.13245) · [MQA](https://arxiv.org/abs/1911.02150) · [PagedAttention](https://arxiv.org/abs/2309.06180) · [StreamingLLM](https://arxiv.org/abs/2309.17453) · [H₂O](https://arxiv.org/abs/2306.14048)
