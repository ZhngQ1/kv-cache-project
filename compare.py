"""Verify paired Needle runs and report exact-match rates by length and depth."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = row["seq_len"], row["depth"], row["trial"]
        if key in rows:
            raise ValueError(f"Duplicate case {key} in {path}")
        rows[key] = row
    return rows


def compare(baseline, quantized):
    if not baseline or baseline.keys() != quantized.keys():
        raise ValueError("Runs must contain the same non-empty set of cases.")
    groups = defaultdict(list)
    expected_pairs = {("transformers", "fp32", "int8"), ("vllm", "auto", "fp8_e4m3")}
    pairs = set()
    for key, before in baseline.items():
        after = quantized[key]
        for field in (
            "backend", "weight_dtype", "device", "model", "revision", "resolved_revision",
            "seed", "max_new_tokens", "expected", "prompt_sha256", "os", "gpu", "torch",
            "transformers", "vllm", "threads", "attention_backend",
        ):
            if before.get(field) != after.get(field):
                raise ValueError(f"Case {key} has different {field}.")
        pairs.add((before["backend"], before["cache"], after["cache"]))
        groups[key[:2]].append((before["correct"], after["correct"]))
    if len(pairs) != 1 or not pairs <= expected_pairs:
        raise ValueError("Use a single supported same-backend cache pair.")
    for (length, depth), group in sorted(groups.items()):
        n = len(group)
        hits_before = sum(before for before, _ in group)
        hits_after = sum(after for _, after in group)
        yield dict(
            seq_len=length, depth=depth, trials=n, baseline_hits=hits_before,
            quantized_hits=hits_after, baseline_accuracy=hits_before / n,
            quantized_accuracy=hits_after / n, delta_pct_points=100 * (hits_after - hits_before) / n,
            lost=sum(before and not after for before, after in group),
            gained=sum(after and not before for before, after in group),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("quantized", type=Path)
    args = parser.parse_args()
    for summary in compare(load(args.baseline), load(args.quantized)):
        print(json.dumps(summary))
