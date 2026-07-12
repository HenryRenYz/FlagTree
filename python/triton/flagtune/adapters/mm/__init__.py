# flagtune/adapters/mm: matrix multiplication operator adapter

from triton.flagtune.registry import OperatorInfo, register

OPERATOR_ID = "mm_general_tma"
_HOPPER_TMA_SMEM_LIMIT_BYTES = 220 * 1024


def _next_power_of_2(value):
    if value <= 1:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _mm_to_config(config_dict):
    from triton import Config

    kwargs = {}
    for field in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_M", "SPLIT_K"):
        if field in config_dict:
            kwargs[field] = int(config_dict[field])

    return Config(
        kwargs=kwargs,
        num_warps=int(config_dict.get("num_warps", 4)),
        num_stages=int(config_dict.get("num_stages", 3)),
    )


def _mm_extract_shape(named_args):
    M = int(named_args.get("M", 0))
    N = int(named_args.get("N", 1 if "M" in named_args and "K" in named_args else 0))
    K = int(named_args.get("K", 0))
    return {
        "M": M,
        "N": N,
        "K": K,
        "stride_am": K,
        "stride_bk": N,
    }


def _mm_select_kernel_variant(shape):
    if int(shape.get("N", 0)) <= 1:
        return "gemv"
    return "mm_general_tma"


def _mm_validate_shape_config(shape, config):
    """Filter configs that are structurally invalid for the mm adapter.

    Hopper TMA kernels are benchmarked through FlagTree AABS
    (auto_adjust_block_sizes). AABS may shrink BLOCK_M/BLOCK_N to match tiny
    runtime dimensions before the TMA pre-hook writes TensorDescriptor
    block_shape. Because of that, a proposer config with BLOCK_M > M or
    BLOCK_N > N is not invalid by itself; the default tuner relies on the same
    mechanism for tiny shapes such as M=4.

    Keep this validator conservative: reject only combinations that select the
    wrong kernel family or are missing core dimensions. Shape-size overshoot is
    handled by AABS plus the per-config TMA pre-hook in FlagGems LibTuner.

    The one resource check below mirrors why some XGB top-ranked candidates
    benchmark as ``inf`` instead of throwing Python exceptions: their adjusted
    A/B TMA tiles exceed Hopper shared-memory capacity once multiplied by
    num_stages. Filtering them before top-k selection lets the proposer pick the
    next valid ranked candidates instead of returning an all-invalid top-k set.
    """
    M = int(shape.get("M", 0))
    N = int(shape.get("N", 0))
    K = int(shape.get("K", 0))
    is_gemv = N <= 1

    if M <= 0 or N <= 0 or K <= 0:
        return False
    if int(config.get("BLOCK_M", 0)) <= 0 or int(config.get("BLOCK_K", 0)) <= 0:
        return False
    if "BLOCK_N" in config:
        bn = int(config.get("BLOCK_N", 0))
        if bn <= 0:
            return False
        if not is_gemv:
            if bn == 1:
                return False
            bm = int(config.get("BLOCK_M", 0))
            bk = int(config.get("BLOCK_K", 0))
            stages = int(config.get("num_stages", 3))
            adjusted_bm = _next_power_of_2(M) if bm > M else bm
            adjusted_bn = _next_power_of_2(N) if bn > N else bn
            adjusted_bk = _next_power_of_2(K) if bk > K else bk
            # TMA matmul stages the A and B tiles in shared memory. The C tile
            # is stored through TMA and is not part of this estimate.
            smem_bytes = (adjusted_bm * adjusted_bk + adjusted_bk * adjusted_bn) * 2 * stages
            if smem_bytes > _HOPPER_TMA_SMEM_LIMIT_BYTES:
                return False
    return True


def _ensure_registered():
    from triton.flagtune.adapters.mm.parameter_space import mm_parameter_space
    from triton.flagtune.adapters.mm.input_space import mm_input_space
    from triton.flagtune.adapters.mm.feature_pipeline import MMFeaturePipeline

    register(
        OperatorInfo(
            operator_id=OPERATOR_ID,
            operator_kind="matmul",
            param_space=mm_parameter_space(),
            input_space=mm_input_space(),
            feature_pipeline=MMFeaturePipeline(),
            to_config=_mm_to_config,
            extract_shape=_mm_extract_shape,
            select_kernel_variant=_mm_select_kernel_variant,
            default_kernel_variant="mm_general_tma",
            op_id="flagtree/gemm",
            validate_shape_config=_mm_validate_shape_config,
        ))

    # Register the same adapter under the FlagGems op_id convention.
    register(
        OperatorInfo(
            operator_id="flaggems_mm_general_tma",
            operator_kind="matmul",
            param_space=mm_parameter_space(),
            input_space=mm_input_space(),
            feature_pipeline=MMFeaturePipeline(),
            to_config=_mm_to_config,
            extract_shape=_mm_extract_shape,
            select_kernel_variant=_mm_select_kernel_variant,
            default_kernel_variant="mm_general_tma",
            op_id="flaggems/mm_general_tma",
            validate_shape_config=_mm_validate_shape_config,
        ))


_ensure_registered()
