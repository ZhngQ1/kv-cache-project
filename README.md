# KV Cache Optimization

This course project measures how grouped-query attention (GQA) changes KV cache growth and how cache quantization affects retrieval from long prompts. CPU measurements and A100 vLLM FP8 retrieval runs have been completed. The team plan below assigns the remaining follow-ups, writing, and presentation work.

## Proposed revised team responsibilities

This proposal accounts for completed work as well as remaining tasks. Replace Teammate B and Teammate C with names after the team agrees. Each member owns an experiment, writes the corresponding paper sections, prepares their presentation material, and speaks for approximately one third of the talk.

### Completed work

Charles has completed the controlled CPU MHA/GQA baseline, the A100 FP16 benchmark with 20 repeats per condition, and the Qwen BF16/FP8 retrieval evaluation. The primary retrieval grid covers seven prompt lengths from 512 to 32,752 tokens, three depths, and 20 trials per cell. Both cache conditions achieve 420/420 exact matches. Experiment setup, result verification, and figure preparation are part of this contribution.

- [A100 MHA/GQA results](results/charles-a100-fp16-20repeats/README.md)
- [Qwen pilot](results/qwen-a100-pilot-20261117/README.md)
- [Primary retrieval grid](results/qwen-a100-full-20261117/README.md)
- [Long-context extension and combined results](results/qwen-a100-long-20261117/README.md)

### Ownership

| Owner | Experimental contribution | Paper contribution | Presentation contribution |
| --- | --- | --- | --- |
| Charles | Own the completed MHA/GQA measurements and shared BF16/FP8 baseline. Finalize memory figures, explain measurement boundaries, and document the existing setup. | Draft the common experimental setup and MHA/GQA methods/results. Cover GQA and MQA in related work. | Explain the research question, KV formula, controlled comparison, and measured memory savings. |
| Teammate B | Independently reproduce a small BF16/FP8 subset, then own retrieval with plausible distractor codes. Implement the revised fixture, run both cache settings, and analyze paired outcomes. | Draft quantized-cache methods/results and the distractor experiment. Cover cache quantization and PagedAttention in related work. | Explain FP8 settings, cache capacity, and retrieval with distractors. |
| Teammate C | Independently reproduce a small BF16/FP8 subset, then own early-position retrieval and missing-answer controls. Implement the controls, run both cache settings, and review scoring and failures. | Draft evaluation details, effective-context analysis, limitations, and a conclusion draft. Cover StreamingLLM and H₂O in related work. | Explain reliability, effective context, the ceiling effect, and limitations. |

### B's follow-up experiment

Use an archive containing several named records with different six-digit codes. Ask for the code of one specified record, and give distractor records the same format as the target. Exactly one record must match the requested name. Verify that the expected code appears only in its target record.

Start with a 512-token pilot at 10%, 50%, and 90% depth using five trials per depth and both BF16 and FP8. Freeze the prompt and scoring after reviewing the pilot. A bounded main grid is 512, 4,096, 16,384, and 32,752 tokens, three depths, and ten fresh trials per cell, giving 120 cases per condition. Keep the existing checkpoint and runtime configuration. Label this as a separate retrieval task.

### C's follow-up experiment

Test the existing single-needle task at earlier positions of 1% and 5%. A bounded main grid is 4,096, 16,384, and 32,752 tokens with ten fresh trials per position, giving 60 cases per condition.

Add a separate missing-answer control with ten cases at each of 512 and 4,096 tokens. Remove the target fact and explicitly instruct the model to output NOT FOUND when the answer is absent. Score exact abstention separately from retrieval accuracy. Run both cache conditions on identical control prompts. Review a small pilot before freezing the wording.

The follow-ups require changes to fixture generation and scoring beyond the current needle.py CLI. Each owner should keep the new task clearly identified and save the exact prompts or token IDs, settings, and scoring code with their results. Full accuracy is a valid outcome. Define the protocol before the main run and report all frozen cells.

### GPU and environment consistency

For each experiment, run the native BF16 and FP8 conditions sequentially on the same GPU in the same Colab allocation, using the same isolated environment. A100-SXM4-40GB is the validated reference GPU. B and C may use different GPU models for their respective accuracy experiments, provided each BF16/FP8 pair stays on one GPU and the hardware is recorded. Run a small paired pilot on any alternative GPU before the main sweep.

If a Colab interruption leads to a different GPU or rebuilt environment, start a new run directory and rerun both cache conditions for that comparison. Preserve earlier results and identify their run separately. Cross-GPU accuracy results should be labeled by hardware. Direct latency and peak-memory comparisons require matching GPU models, software, workloads, and measurement settings.

Use the following reference configuration for B's and C's retrieval experiments.

| Setting | Validated reference |
| --- | --- |
| Model | Qwen/Qwen2.5-0.5B-Instruct |
| Model and tokenizer revision | 7ae557604adf67be50417f59c2c2f167def9a775 |
| Python | 3.12, in the isolated qwen-env environment |
| vLLM | 0.10.1 |
| PyTorch build | 2.7.1+cu126 |
| Transformers | 4.57.1 |
| Engine | V0, with VLLM_USE_V1=0 |
| Attention backend | XFormers, with VLLM_ATTENTION_BACKEND=XFORMERS |
| Model weight dtype in both conditions | bfloat16 |
| Native KV cache | auto, following the BF16 model dtype |
| Quantized KV cache | fp8_e4m3, with calculate_kv_scales=True |
| Generation | Greedy, temperature=0, maximum 16 output tokens |
| Execution | enforce_eager=True, enable_prefix_caching=False, max_num_seqs=1 |
| GPU memory utilization | 0.90 in both conditions |
| Engine sequence limit | Same for both conditions in a sweep, covering the longest prompt plus 16 output tokens |

Within a BF16/FP8 pair, keep the prompt token IDs, trial seeds, context lengths, insertion depths, generation settings, scoring rules, and request order identical. The intended treatment is KV cache precision and its associated FP8 scale calculation. Each follow-up may define its own prompt and seed set. Freeze these after its pilot and use fresh seeds for its main evaluation. For the 32,768-token context budget, the longest prompt is 32,752 tokens when reserving 16 output tokens.

Keep the GPU retrieval environment separate from the CPU requirements.txt environment. Save the following with every run.

- settings.json with GPU model, model/tokenizer SHA, task version, seeds, grid, engine options, cache dtype, and scale policy.
- The output of nvidia-smi, Python version, and pip freeze from the isolated environment, including the PyTorch CUDA build.
- The exact evaluation script version, prompts or token IDs, per-case answers and scores, and input hashes.
- Complete startup logs for both conditions, showing the resolved model/tokenizer revisions, weight dtype, engine, attention backend, KV dtype, and cache capacity.
- The paired comparison summary. Use compare.py for its supported checks and review settings and startup logs for the remaining configuration details.

### Review and integration

- Charles reviews B's experiment configuration and paired results.
- B reviews C's prompt construction, missing-answer controls, and scoring.
- C reviews Charles's memory calculations, figure labels, and written claims.
- Each owner contributes their own figures, methods text, result interpretation, and presentation material.
- All three review the abstract and final conclusion, verify references, and rehearse transitions. Rotate the final integration work across the paper, figures, and presentation.

### Milestones

| Date | Deliverable |
| --- | --- |
| Nov 18 | Agree on names and ownership. B and C reproduce a small subset and freeze follow-up pilots. Charles shares the completed setup and result notes. |
| Nov 19–24 | Complete the bounded follow-ups and each owner's first methods/results draft. |
| Nov 25–28 | Cross-review data, scoring, figures, and related work. |
| Nov 29–30 | Finalize presentation material and rehearse with approximately equal speaking time. |
| Dec 3 | Course presentation, using the planning date from the prior course discussion. |
| Dec 4–9 | Address feedback, finish paper integration, and complete a joint reproducibility check. |
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

The CPU INT8 implementation includes scale bytes in the stored-cache measurement. It dequantizes before attention and creates temporary tensors. Treat the payload measurement and retrieval accuracy as the reported outcomes. The offline smoke test verifies both code paths with random weights. Pretrained CPU FP32/INT8 retrieval accuracy remains unmeasured. The completed GPU BF16/FP8 results are linked above.

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

Use `--gpu-dtype float16` for both conditions if the GPU requires it. Keep both runs on the same GPU, with identical model weights, prompts, vLLM version, and generation settings. Record the resolved cache dtype, scale policy, attention backend, and KV capacity from the logs. If the backend changes with the cache setting, report the result as a comparison of two vLLM configurations. Distinguish theoretical payload and actual cache capacity from total GPU allocation, which can include a preallocated cache pool. GPU FP8 has been verified on A100 using vLLM 0.10.1, V0, and XFormers, with BF16 model weights. See the result links above. The commands in this section describe the pilot interface; the completed Colab runs also set VLLM_USE_V1=0 and VLLM_ATTENTION_BACKEND=XFORMERS in the child-process environment.

## Related work

Cover cache compression, sparse access and eviction, and architectural simplification. Discuss PagedAttention under cache allocation and sharing. [GQA](https://arxiv.org/abs/2305.13245) · [MQA](https://arxiv.org/abs/1911.02150) · [PagedAttention](https://arxiv.org/abs/2309.06180) · [StreamingLLM](https://arxiv.org/abs/2309.17453) · [H₂O](https://arxiv.org/abs/2306.14048)
