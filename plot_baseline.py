import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Plot the 36-row CPU baseline sweep.")
parser.add_argument("results_dir", type=Path)
args = parser.parse_args()
root = args.results_dir
rows = [json.loads(line) for line in (root / "baseline.jsonl").read_text().splitlines()]
assert len(rows) == 36
groups = defaultdict(list)
for row in rows:
    assert row["kv_bytes"] == row["predicted_kv_bytes"]
    groups[row["architecture"], row["seq_len"]].append(row)
summary = []
for (architecture, length), samples in sorted(groups.items()):
    assert len(samples) == 3
    summary.append(dict(
        architecture=architecture, seq_len=length, repeats=3,
        kv_bytes=samples[0]["kv_bytes"], kv_gb=samples[0]["kv_gb"],
        prefill_ms_mean=mean(x["prefill_ms"] for x in samples),
        prefill_ms_std=stdev(x["prefill_ms"] for x in samples),
    ))
with (root / "summary.csv").open("w") as output:
    writer = csv.DictWriter(output, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)
for length in sorted({row["seq_len"] for row in rows}):
    mha, gqa = groups["MHA", length][0], groups["GQA", length][0]
    assert mha["kv_bytes"] == 4 * gqa["kv_bytes"]
fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
for architecture, color, heads in [("MHA", "#2457A7", 8), ("GQA", "#D16925", 2)]:
    samples = sorted((row for row in summary if row["architecture"] == architecture),
                     key=lambda row: row["seq_len"])
    ax.plot([row["seq_len"] for row in samples], [row["kv_gb"] for row in samples],
            marker="o", color=color, linewidth=2, label=f"{architecture} · {heads} KV heads")
ax.set(xlabel="Prompt length (tokens)", ylabel="Measured KV cache payload (GB)",
       title="KV cache growth with prompt length", ylim=(0, 0.075))
ax.set_xticks([0, 2048, 4096, 6144, 8192])
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.text(0.15, -0.04, "Random Llama · 4 layers · 8 Q heads · head dim 32 · batch 1 · CPU FP32\n"
         "Three repeats overlap exactly. 1 GB = 10⁹ bytes.", fontsize=9)
for extension in ("png", "svg"):
    fig.savefig(root / f"kv-cache-growth.{extension}", dpi=220, bbox_inches="tight")
print(json.dumps({"rows": len(rows), "all_bytes_match_formula": True,
                  "mha_bytes_per_token": 8192, "gqa_bytes_per_token": 2048,
                  "kv_reduction_percent": 75}))
