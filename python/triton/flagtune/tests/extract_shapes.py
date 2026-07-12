#!/usr/bin/env python3
"""
Extract unique (M,N,K) mm shapes from shape-config/*.yaml files,
then sample a diverse ~45-shape subset for benchmarking.
"""
import sys
import re
import math
from pathlib import Path


def extract_shapes(shape_config_dir):
    shapes = set()
    for yf in sorted(Path(shape_config_dir).glob("*.yaml")):
        if "_count" in yf.name or "_gain" in yf.name or "_lose" in yf.name:
            continue
        with open(yf) as f:
            lines = f.readlines()
        in_mm_shapes = False
        buf = []
        for line in lines:
            stripped = line.strip()
            if stripped == "mm:":
                in_mm_shapes = False
                continue
            if stripped == "shapes:":
                in_mm_shapes = True
                buf = []
                continue
            if not in_mm_shapes: continue
            if not stripped.startswith("-"):
                if len(buf) == 4:
                    shapes.add(tuple(int(v) for v in buf))
                    buf = []
                in_mm_shapes = False
                continue
            m = re.match(r'-\s*-?\s*(\d+)', stripped)
            if m:
                buf.append(m.group(1))
                if len(buf) == 4:
                    _, m_val, n, k = int(buf[0]), int(buf[1]), int(buf[2]), int(buf[3])
                    shapes.add((m_val, n, k))
                    buf = []
            elif stripped == '-':
                continue
            else:
                buf = []
    return sorted(shapes, key=lambda x: x[0] * x[1] * x[2])


def ops(s):
    return s[0] * s[1] * s[2]


def log10ops(s):
    v = ops(s)
    return math.log10(v) if v > 0 else 0


def aspect(s):
    m, n, k = s
    if n == 0: return float('inf')
    return m / n


def sample_diverse(shapes, target=45):
    """Diverse sampling across scales and types."""
    selected = set()

    # 1. GEMV: pick ~6 (M=1 or N=1, across scales)
    gemv = [s for s in shapes if s[0] == 1 or s[1] == 1]
    gemv_by_ops = sorted(gemv, key=ops)
    step = max(1, len(gemv_by_ops) // 6)
    for i in range(0, len(gemv_by_ops), step):
        selected.add(gemv_by_ops[i])
        if len([x for x in selected if x[0] == 1 or x[1] == 1]) >= 6: break

    # 2. Bucket by log10(ops) and pick 2-3 per bucket
    buckets = {}
    for s in shapes:
        b = int(log10ops(s)) if ops(s) > 0 else 0
        buckets.setdefault(b, []).append(s)

    for scale in sorted(buckets):
        bucket = buckets[scale]
        # Sort by aspect ratio uniqueness, pick evenly
        bucket_sorted = sorted(bucket, key=lambda s: (s[2], abs(aspect(s) - 1)))
        n = min(3, max(1, target * len(bucket) // len(shapes)))
        step = max(1, len(bucket_sorted) // n)
        for i in range(0, len(bucket_sorted), step):
            if len(selected) >= target: break
            selected.add(bucket_sorted[i])
        if len(selected) >= target: break

    # 3. Ensure coverage of key K values
    key_k = {1024, 1152, 1536, 2048, 2304, 4096, 4304, 4608, 7168, 9216, 12288, 20480}
    for k_val in key_k:
        if len(selected) >= target: break
        candidates = [s for s in shapes if s[2] == k_val and s not in selected and s[0] > 1 and s[1] > 1]
        if candidates:
            selected.add(sorted(candidates, key=lambda s: abs(aspect(s) - 1))[0])

    result = sorted(selected, key=ops)[:target]
    return result


if __name__ == "__main__":
    shape_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    all_shapes = extract_shapes(shape_dir)
    print(f"# Total unique mm shapes: {len(all_shapes)}")
    sampled = sample_diverse(all_shapes, target=45)
    print(f"# Sampled {len(sampled)} diverse shapes\n")
    print("SHAPES = [")
    for s in sampled:
        print(f"    ({s[0]}, {s[1]}, {s[2]}),")
    print("]")
