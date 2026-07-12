"""
Matrix multiplication (mm) operator kernel adapter — LibTuner version.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, Dict, List, Optional, Tuple

from triton.flagtune.core.interfaces import KernelAdapter

_KERNEL_ATTR_MAP: Dict[str, str] = {
    "gemv": "gemv_kernel",
    "mm_general_tma": "mm_kernel_general_host_tma",
}

_CONFIG_MAKER_MAP: Dict[str, str] = {
    "gemv": "_make_gemv_configs",
    "mm_general_tma": "_make_mm_tma_configs",
}

_KNOWN_MM_MODULES: List[str] = [
    "flag_gems.runtime.backend._nvidia.hopper.ops.mm",
    "flag_gems.ops.mm",
]


class MMLibTunerAdapter(KernelAdapter):

    def __init__(self, torch_module: Any = None) -> None:
        self._torch = torch_module

    @property
    def torch(self) -> Any:
        if self._torch is None:
            import torch as _torch
            self._torch = _torch
        return self._torch

    def make_bench_args(
        self,
        shape: Dict[str, Any],
        config: Dict[str, Any],
        kernel_variant: str,
    ) -> Tuple[tuple, Dict[str, Any]]:
        torch = self.torch
        m = int(shape.get("M", 1))
        n = int(shape.get("N", 1))
        k = int(shape.get("K", 1))
        a = shape.get("_a")
        b = shape.get("_b")
        if a is None or b is None:
            a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
            b = torch.randn((k, n), device="cuda", dtype=torch.bfloat16)
        if kernel_variant == "gemv":
            return self._make_gemv_args(a, b, m, k, torch)
        else:
            return self._make_tma_args(a, b, m, n, k, torch)

    def find_tuner(self, kernel_variant: str) -> Optional[Any]:
        attr_name = _KERNEL_ATTR_MAP.get(kernel_variant)
        if attr_name is None:
            return None
        for module_name, module in list(sys.modules.items()):
            kernel = getattr(module, attr_name, None)
            if kernel is None:
                continue
            tuner = self._unwrap_tuner(kernel)
            if tuner is not None:
                return tuner
        for mod_path in _KNOWN_MM_MODULES:
            try:
                module = importlib.import_module(mod_path)
                kernel = getattr(module, attr_name, None)
                if kernel is not None:
                    tuner = self._unwrap_tuner(kernel)
                    if tuner is not None:
                        return tuner
            except ImportError:
                continue
        return None

    def make_config_objects(
        self,
        entries: List[Dict[str, Any]],
        kernel_variant: str,
    ) -> List[Any]:
        maker_name = _CONFIG_MAKER_MAP.get(kernel_variant)
        if maker_name is None:
            return []
        for mod_path in _KNOWN_MM_MODULES:
            try:
                module = importlib.import_module(mod_path)
                maker = getattr(module, maker_name, None)
                if maker is not None:
                    return maker(entries)
            except ImportError:
                continue
        return []

    def install_configs(self, tuner: Any, configs: List[Any]) -> None:
        if hasattr(tuner, "_flagtune_runtime_configs"):
            tuner._flagtune_runtime_configs = configs

    @staticmethod
    def _unwrap_tuner(kernel: Any) -> Optional[Any]:
        if hasattr(kernel, "_flagtune_default_configs"):
            return kernel
        if hasattr(kernel, "fn"):
            inner = kernel.fn
            if hasattr(inner, "_flagtune_default_configs"):
                return inner
        if hasattr(kernel, "__wrapped__"):
            inner = kernel.__wrapped__
            if hasattr(inner, "_flagtune_default_configs"):
                return inner
        return None

    def _make_gemv_args(self, a: Any, b: Any, m: int, k: int, torch: Any) -> Tuple[tuple, Dict[str, Any]]:
        import triton
        c_dtype = self._higher_dtype(torch, a.dtype, b.dtype)
        c = torch.empty((m, 1), device=a.device, dtype=c_dtype)

        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_M"]), )

        return (
            a,
            b,
            c,
            m,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
        ), {
            "grid": grid,
            "IS_FP64": a.dtype == torch.float64,
            "warmup": False,
        }

    def _make_tma_args(self, a: Any, b: Any, m: int, n: int, k: int, torch: Any) -> Tuple[tuple, Dict[str, Any]]:
        import triton
        from triton.tools.tensor_descriptor import TensorDescriptor

        a_row_major = a.stride(1) == 1
        b_row_major = b.stride(1) == 1
        dummy_block = [1, 1]
        c_dtype = self._higher_dtype(torch, a.dtype, b.dtype)
        c = torch.empty((m, n), device=a.device, dtype=c_dtype)
        if a_row_major:
            a_desc = TensorDescriptor(a, a.shape, a.stride(), dummy_block)
        else:
            a_desc = TensorDescriptor(a, a.T.shape, a.T.stride(), dummy_block)
        if b_row_major:
            b_desc = TensorDescriptor(b, b.shape, b.stride(), dummy_block)
        else:
            b_desc = TensorDescriptor(b, b.T.shape, b.T.stride(), dummy_block)
        c_desc = TensorDescriptor(c, c.shape, c.stride(), dummy_block)
        dtype_str = str(a.dtype).split(".")[-1]

        def grid(meta):
            return (triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]), )

        return (
            a_desc,
            b_desc,
            c_desc,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        ), {
            "grid": grid,
            "A_ROW_MAJOR": a_row_major,
            "B_ROW_MAJOR": b_row_major,
            "dtype": dtype_str,
            "warmup": False,
        }

    @staticmethod
    def _higher_dtype(torch: Any, left: Any, right: Any) -> Any:
        ordered = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
        if left == right:
            return left
        li = ordered.index(left) if left in ordered else -1
        ri = ordered.index(right) if right in ordered else -1
        return left if li >= ri else right
