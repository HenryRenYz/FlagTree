#!/usr/bin/env python
"""
Benchmark: FlagTree default autotune vs FlagTune-predicted configs.

Compares matmul kernel performance (TFLOPS) with and without FlagTune
across diverse matrix shapes.

Usage:
    # Default autotune (baseline)
    TRITON_USE_FLAGTUNE=0 python test/flagtune/test_bench_matmul.py

    # FlagTune-predicted configs
    TRITON_USE_FLAGTUNE=1 TRITON_FLAGTUNE_MODEL_DIR=/path/to/model \
        python test/flagtune/test_bench_matmul.py

    # Automatically run both modes and compare (recommended)
    python test/flagtune/test_bench_matmul.py --compare
"""

import argparse
import gc
import os

import torch
import triton
import triton.ops

SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (16, 4096, 4096),
    (4096, 16, 4096),
    (4096, 4096, 16),
    (512, 16384, 512),
    (16384, 512, 512),
    (1, 4096, 4096),
    (4096, 1, 4096),
    (128, 128, 128),
    (1920, 1920, 1920),
    (3072, 768, 768),
    (768, 3072, 768),
    (768, 768, 3072),
    (11008, 4096, 4096),
    (4096, 11008, 4096),
]

WARMUP = 25
REP = 100


def clear_autotune_cache():
    kernel = triton.ops.matmul._kernel
    kernel.cache = {}
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def bench_matmul(M, N, K, dtype=torch.float16):
    """Benchmark matmul latency (ms) for given shape.  Returns p50 ms."""
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype)

    clear_autotune_cache()

    ms = triton.testing.do_bench(
        lambda: triton.ops.matmul(a, b),
        warmup=WARMUP,
        rep=REP,
    )
    if isinstance(ms, (list, tuple)):
        ms = ms[0]
    tflops = (2.0 * M * N * K) / (float(ms) * 1e-3) / 1e12
    return float(ms), tflops


def run_comparison(model_dir=None):
    if model_dir:
        os.environ["TRITON_FLAGTUNE_MODEL_DIR"] = model_dir

    header = f"{'Shape':<24} {'Default (ms)':>12} {'FlagTune (ms)':>12} {'Speedup':>10} {'Def TFLOPS':>12} {'FT TFLOPS':>12}"
    print(header)
    print("-" * len(header))

    default_failed = 0
    flagtune_failed = 0

    for M, N, K in SHAPES:
        shape_str = f"({M},{N},{K})"

        # Default autotune
        os.environ["TRITON_USE_FLAGTUNE"] = "0"
        try:
            ms_def, tflops_def = bench_matmul(M, N, K)
        except Exception:
            ms_def, tflops_def = 999.99, 0.0
            default_failed += 1

        # FlagTune
        os.environ["TRITON_USE_FLAGTUNE"] = "1"
        try:
            ms_ft, tflops_ft = bench_matmul(M, N, K)
        except Exception:
            ms_ft, tflops_ft = 999.99, 0.0
            flagtune_failed += 1

        speedup = ms_def / ms_ft if ms_ft > 0 else 0.0
        print(f"{shape_str:<24} {ms_def:>12.4f} {ms_ft:>12.4f} {speedup:>9.2f}x {tflops_def:>12.2f} {tflops_ft:>12.2f}")

    print("-" * len(header))
    print(f"Default errors: {default_failed}, FlagTune errors: {flagtune_failed}")


def run_single_mode(mode="default", model_dir=None):
    if model_dir:
        os.environ["TRITON_FLAGTUNE_MODEL_DIR"] = model_dir
    os.environ["TRITON_USE_FLAGTUNE"] = "1" if mode == "flagtune" else "0"

    print(f"Mode: {mode.upper()}")
    print(f"{'Shape':<24} {'Latency (ms)':>12} {'TFLOPS':>12}")
    print("-" * 50)

    for M, N, K in SHAPES:
        shape_str = f"({M},{N},{K})"
        try:
            ms, tflops = bench_matmul(M, N, K)
            print(f"{shape_str:<24} {ms:>12.4f} {tflops:>12.2f}")
        except Exception as e:
            print(f"{shape_str:<24} {'ERR':>12} {str(e)[:30]}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark matmul: default autotune vs FlagTune")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both modes sequentially and show speedup comparison",
    )
    parser.add_argument(
        "--mode",
        choices=["default", "flagtune"],
        default="default",
        help="Run only one mode (default, flagtune)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Path to trained XGBoost model directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of top configs per shape (overrides TRITON_FLAGTUNE_TOP_K)",
    )
    args = parser.parse_args()

    if args.top_k is not None:
        os.environ["TRITON_FLAGTUNE_TOP_K"] = str(args.top_k)

    if args.compare:
        run_comparison(model_dir=args.model_dir)
    else:
        run_single_mode(mode=args.mode, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
