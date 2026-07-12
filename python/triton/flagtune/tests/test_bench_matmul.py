#!/usr/bin/env python
"""
Benchmark: @flagtune decorator on 34 diverse matmul shapes.

Usage:
    TRITON_USE_FLAGTUNE=1 TRITON_FLAGTUNE_MODEL_DIR=/path/to/model python test_bench_matmul.py
    FLAGTUNE_DISABLE_OPS=flagtree/gemm python test_bench_matmul.py  # test disable
"""

import argparse
import gc
import os
import statistics
import torch
import triton
import triton.language as tl
from triton.flagtune import flagtune

DTYPE = torch.float16
TRIALS = 3
WARMUP = 25
REP = 100

SHAPES = [
    (2048, 248320, 2048),
    (12675, 7168, 4608),
    (65536, 1152, 4304),
    (65536, 4304, 1152),
    (8276, 7168, 2304),
    (14429, 7168, 1024),
    (12296, 1536, 1536),
    (4107, 4096, 1024),
    (65536, 1152, 144),
    (3076, 3072, 1536),
    (272, 12288, 2048),
    (99, 31040, 4096),
    (33, 31040, 4096),
    (4107, 4096, 128),
    (1036, 4096, 128),
    (184, 4096, 128),
    (32, 3072, 1536),
    (16, 2048, 4096),
    (208, 64, 7168),
    (40, 1024, 2048),
    (64, 64, 7168),
    (224, 16, 4096),
    (4107, 1, 2048),
    (16, 4096, 128),
    (4, 512, 4096),
    (368, 1, 4096),
    (16, 16, 4096),
    (400, 1, 2048),
    (1, 4096, 128),
    (64, 64, 2048),
    (112, 1, 4096),
    (96, 1, 2048),
    (8, 1, 2048),
    (1, 1, 2048),
]


def get_configs():
    configs = []
    for block_m in [64, 128, 256]:
        for block_n in [64, 128, 256]:
            for block_k in [32, 64, 128]:
                for num_warps in [4, 8]:
                    for num_stages in [2, 3, 4]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k, "SPLIT_K": 1},
                                num_warps=num_warps,
                                num_stages=num_stages,
                            ))
    return configs


def _smem_filter(configs, named_args):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    max_smem = props.shared_memory_per_block_optin
    dtsize = named_args["A"].element_size()
    pruned = []
    for cfg in configs:
        kw = cfg.kwargs
        smem = (kw["BLOCK_M"] + kw["BLOCK_N"]) * kw["BLOCK_K"] * cfg.num_stages * dtsize
        if smem <= max_smem:
            pruned.append(cfg)
    return pruned if pruned else configs


@flagtune(
    configs=get_configs(),
    key=["M", "N", "K"],
    flagtune_op_id="flagtree/gemm",
    prune_configs_by={
        "early_config_prune":
        lambda c, na, **kw: _smem_filter(c, na),
        "perf_model":
        lambda BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, M, N, K, SPLIT_K, **kw: (M * N * K) /
        (BLOCK_M * BLOCK_N * BLOCK_K),
        "top_k":
        10,
    },
)
@triton.heuristics({"EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0})
@triton.jit
def _matmul_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_M: tl.constexpr = 8,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A_ptr = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    if BLOCK_N == 1:
        # GEMV path: element-wise multiply + reduce (tl.dot unsupported for N=1 on Hopper)
        B_vec = B + rk * stride_bk  # 1D pointer: shape (BLOCK_K,), compatible with a * b broadcast
        acc = tl.zeros((BLOCK_M, ), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                mr = M - pid_m * BLOCK_M
                a = tl.load(A_ptr, mask=rm[:, None] < mr, other=0.0)
                b = tl.load(B_vec)
            else:
                kr = K - k * (BLOCK_K * SPLIT_K)
                a = tl.load(A_ptr, mask=rk[None, :] < kr, other=0.0)
                b = tl.load(B_vec, mask=rk < kr, other=0.0)
            acc += tl.sum(a * b, axis=1)
            A_ptr += BLOCK_K * SPLIT_K * stride_ak
            B_vec += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C.dtype.element_ty)
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rm_s = tl.where(rm2 < M, rm2, 0)
        C_ptr = C + rm_s * stride_cm
        tl.store(C_ptr, acc, mask=rm2 < M)
    else:
        # GEMM path: tl.dot for N >= 2
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
            if EVEN_K:
                mr = M - pid_m * BLOCK_M
                nr = N - pid_n * BLOCK_N
                a = tl.load(A_ptr, mask=rm[:, None] < mr, other=0.0)
                b = tl.load(B_ptr, mask=rn[None, :] < nr, other=0.0)
            else:
                kr = K - k * (BLOCK_K * SPLIT_K)
                a = tl.load(A_ptr, mask=rk[None, :] < kr, other=0.0)
                b = tl.load(B_ptr, mask=rk[:, None] < kr, other=0.0)
            acc += tl.dot(a, b)
            A_ptr += BLOCK_K * SPLIT_K * stride_ak
            B_ptr += BLOCK_K * SPLIT_K * stride_bk
        acc = acc.to(C.dtype.element_ty)
        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rm_s = tl.where(rm2 < M, rm2, 0)
        rn_s = tl.where(rn2 < N, rn2, 0)
        C_ptr = C + (rm_s[:, None] * stride_cm + rn_s[None, :] * stride_cn)
        tl.store(C_ptr, acc, mask=(rm2 < M)[:, None] & (rn2 < N)[None, :])


def matmul(a, b):
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]), )

    _matmul_kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


def run_benchmark(trials=TRIALS):
    from triton.flagtune import is_enabled
    ft_active = is_enabled()
    ft_disabled_ops = os.environ.get("FLAGTUNE_DISABLE_OPS", "").strip()
    print(f"FlagTune enabled: {ft_active}")
    if ft_disabled_ops:
        print(f"FLAGTUNE_DISABLE_OPS: {ft_disabled_ops}")

    hdr = f"{'Shape':<22} {'Latency ms (+-s)':>18} {'TFLOPS':>10} {'Config'}"
    print(hdr)
    print("-" * 80)

    for M, N, K in SHAPES:
        s = f"({M},{N},{K})"
        try:
            a = torch.randn((M, K), device="cuda", dtype=DTYPE)
            b = torch.randn((K, N), device="cuda", dtype=DTYPE)
            _matmul_kernel.cache = {}
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            lats = []
            for t in range(trials):
                if t > 0:
                    _matmul_kernel.cache = {}
                    torch.cuda.empty_cache()
                ms = triton.testing.do_bench(lambda: matmul(a, b), warmup=WARMUP, rep=REP)
                if isinstance(ms, (list, tuple)):
                    ms = ms[0]
                lats.append(float(ms))

            mean_ms = statistics.mean(lats)
            std_ms = statistics.stdev(lats) if len(lats) >= 2 else 0
            tflops = (2.0 * M * N * K) / (mean_ms * 1e-3) / 1e12

            key = (int(M), int(N), int(K), str(DTYPE), str(DTYPE), str(DTYPE))
            cfg = _matmul_kernel.cache.get(key)
            cfg_str = f"M{cfg.kwargs.get('BLOCK_M','?')}/N{cfg.kwargs.get('BLOCK_N','?')}/K{cfg.kwargs.get('BLOCK_K','?')}/w{cfg.num_warps}/s{cfg.num_stages}" if cfg else "N/A"

            print(f"{s:<22} {mean_ms:>8.4f}+-{std_ms:<7.4f} {tflops:>10.2f} {cfg_str}")

            a = b = None
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"{s:<22} {'FAIL':>18} {str(e)[:40]}")

    print("-" * 80)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=TRIALS)
    args = p.parse_args()
    run_benchmark(trials=args.trials)


if __name__ == "__main__":
    main()
