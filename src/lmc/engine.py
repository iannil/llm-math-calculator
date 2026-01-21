"""Core computation engine based on Megatron-LM formulas."""

from typing import Optional

from .loader import get_hardware
from .models import (
    ComputeResult,
    HardwareSpec,
    MemoryBreakdown,
    ParallelismConfig,
    TrainingConfig,
    ZeROStage,
)


def calc_total_flops(params: int, tokens: int, is_moe: bool = False) -> float:
    """
    Calculate total training FLOPs.

    Formula: 6 * P * D (forward + backward pass)
    For MoE models, use active_params instead of total_params.

    Args:
        params: Number of parameters (use active_params for MoE)
        tokens: Number of training tokens
        is_moe: Whether this is an MoE model

    Returns:
        Total FLOPs for training
    """
    return 6.0 * params * tokens


def calc_memory_model_states(
    params: int,
    zero_stage: ZeROStage = ZeROStage.ZERO_0,
    num_gpus: int = 1,
    use_fp16: bool = True,
) -> float:
    """
    Calculate memory for model states (parameters + gradients + optimizer states).

    ZeRO-0 (no ZeRO): 16 bytes per parameter
      - Parameters: 2 bytes (fp16) + 2 bytes (fp32 master copy)
      - Gradients: 2 bytes (fp16)
      - Optimizer: 4 bytes (momentum) + 4 bytes (variance) + 2 bytes (padding)

    ZeRO-1: Optimizer states partitioned across GPUs
    ZeRO-2: Optimizer states + gradients partitioned
    ZeRO-3: Everything partitioned

    Args:
        params: Number of parameters
        zero_stage: ZeRO optimization stage
        num_gpus: Number of GPUs for partitioning
        use_fp16: Whether using mixed precision

    Returns:
        Memory in bytes
    """
    if use_fp16:
        # Mixed precision: fp16 params + fp32 master weights + fp16 grads + fp32 optimizer
        param_bytes = 2  # fp16 parameters
        master_bytes = 4  # fp32 master weights
        grad_bytes = 2  # fp16 gradients
        optim_bytes = 8  # fp32 momentum + variance (Adam)
        total_per_param = param_bytes + master_bytes + grad_bytes + optim_bytes  # 16 bytes
    else:
        # Full precision
        param_bytes = 4
        grad_bytes = 4
        optim_bytes = 8
        total_per_param = param_bytes + grad_bytes + optim_bytes  # 16 bytes

    base_memory = params * total_per_param

    if zero_stage == ZeROStage.ZERO_0:
        return float(base_memory)
    elif zero_stage == ZeROStage.ZERO_1:
        # Optimizer states partitioned (8 bytes)
        partitioned = params * optim_bytes / num_gpus
        non_partitioned = params * (param_bytes + master_bytes + grad_bytes)
        return float(partitioned + non_partitioned)
    elif zero_stage == ZeROStage.ZERO_2:
        # Optimizer states + gradients partitioned
        partitioned = params * (optim_bytes + grad_bytes) / num_gpus
        non_partitioned = params * (param_bytes + master_bytes)
        return float(partitioned + non_partitioned)
    else:  # ZeRO-3
        # Everything partitioned
        return float(base_memory / num_gpus)


def calc_memory_activations(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    num_attention_heads: int,
    use_recomputation: bool = False,
    tensor_parallel: int = 1,
) -> float:
    """
    Calculate activation memory per micro-batch.

    Based on Megatron-LM formula:
    memory = seq * batch * hidden * layers * (34 + 5 * heads * seq / hidden)

    With recomputation (activation checkpointing), memory is reduced significantly.

    Args:
        batch_size: Micro batch size
        seq_length: Sequence length
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        use_recomputation: Whether using activation checkpointing
        tensor_parallel: Tensor parallelism degree

    Returns:
        Memory in bytes
    """
    s = seq_length
    b = batch_size
    h = hidden_size
    L = num_layers
    a = num_attention_heads

    # Attention score memory term
    attention_term = 5.0 * a * s / h

    # Total activation memory per layer (bytes, assuming fp16 = 2 bytes)
    bytes_per_element = 2
    memory_per_layer = s * b * h * (34 + attention_term) * bytes_per_element

    total_memory = memory_per_layer * L

    # Tensor parallelism reduces activation memory
    total_memory /= tensor_parallel

    # Activation checkpointing reduces memory by sqrt(L) factor approximately
    if use_recomputation:
        import math
        total_memory /= math.sqrt(L)

    return float(total_memory)


def calc_memory_kv_cache(
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    num_layers: int,
    num_kv_heads: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
) -> float:
    """
    Calculate KV cache memory for inference.

    Formula: 2 * batch * seq * hidden * layers * 2 bytes
    For GQA/MQA: adjust by num_kv_heads / num_attention_heads ratio

    Args:
        batch_size: Batch size
        seq_length: Sequence length
        hidden_size: Hidden dimension
        num_layers: Number of transformer layers
        num_kv_heads: Number of KV heads (for GQA/MQA)
        num_attention_heads: Number of attention heads

    Returns:
        Memory in bytes
    """
    bytes_per_element = 2  # fp16

    # Base KV cache: 2 (K and V) * batch * seq * hidden * layers
    base_memory = 2 * batch_size * seq_length * hidden_size * num_layers * bytes_per_element

    # Adjust for GQA/MQA if applicable
    if num_kv_heads is not None and num_attention_heads is not None:
        kv_ratio = num_kv_heads / num_attention_heads
        base_memory *= kv_ratio

    return float(base_memory)


def calc_training_time(
    total_flops: float,
    num_gpus: int,
    peak_tflops: float,
    mfu: float,
) -> float:
    """
    Calculate training time in seconds.

    Formula: Total FLOPs / (num_GPUs * peak_TFLOPS * 1e12 * MFU)

    Args:
        total_flops: Total training FLOPs
        num_gpus: Number of GPUs
        peak_tflops: Peak TFLOPS per GPU (fp16/bf16)
        mfu: Model FLOPS Utilization (0-1)

    Returns:
        Training time in seconds
    """
    effective_tflops = num_gpus * peak_tflops * 1e12 * mfu
    return total_flops / effective_tflops


def format_flops(flops: float) -> str:
    """Format FLOPs to human-readable string."""
    if flops >= 1e24:
        return f"{flops / 1e24:.2f} YFLOPs"
    elif flops >= 1e21:
        return f"{flops / 1e21:.2f} ZFLOPs"
    elif flops >= 1e18:
        return f"{flops / 1e18:.2f} EFLOPs"
    elif flops >= 1e15:
        return f"{flops / 1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    else:
        return f"{flops / 1e9:.2f} GFLOPs"


def format_time(seconds: float) -> str:
    """Format time duration to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    else:
        days = seconds / 86400
        if days < 30:
            return f"{days:.1f} days"
        elif days < 365:
            return f"{days / 30:.1f} months"
        else:
            return f"{days / 365:.1f} years"


def format_bytes(bytes_val: float) -> str:
    """Format bytes to human-readable string."""
    if bytes_val >= 1e12:
        return f"{bytes_val / 1e12:.2f} TB"
    elif bytes_val >= 1e9:
        return f"{bytes_val / 1e9:.2f} GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val / 1e6:.2f} MB"
    else:
        return f"{bytes_val / 1e3:.2f} KB"


def estimate_resources(config: TrainingConfig) -> ComputeResult:
    """
    Estimate complete training resources.

    Args:
        config: Training configuration

    Returns:
        Complete resource estimation result
    """
    from .optimizer import recommend_parallelism

    warnings: list[str] = []
    notes: list[str] = []

    # Get hardware specs
    hardware = get_hardware(config.gpu_name)
    if hardware is None:
        raise ValueError(f"Unknown GPU: {config.gpu_name}")

    # Use config MFU or hardware typical MFU
    mfu = config.mfu or hardware.typical_mfu

    # Calculate total FLOPs
    total_flops = calc_total_flops(
        params=config.active_params,
        tokens=config.tokens,
        is_moe=config.active_params_billion is not None,
    )

    # Get parallelism recommendation
    parallelism = recommend_parallelism(
        params_billion=config.params_billion,
        memory_gb=hardware.memory_gb,
        num_gpus=config.num_gpus,
        has_nvlink=hardware.nvlink_bandwidth_gbps > 0,
    )

    # Override with config values if specified
    tp = config.tensor_parallel if config.tensor_parallel > 1 else parallelism.tensor_parallel
    pp = config.pipeline_parallel if config.pipeline_parallel > 1 else parallelism.pipeline_parallel
    dp = config.num_gpus // (tp * pp)

    zero_stage = config.zero_stage

    # Calculate memory components
    model_states_bytes = calc_memory_model_states(
        params=config.params,
        zero_stage=zero_stage,
        num_gpus=dp,  # ZeRO works within data parallel group
        use_fp16=config.use_fp16,
    )

    # For memory calculation, use defaults if not specified
    hidden = config.hidden_size
    layers = config.num_layers
    heads = config.num_attention_heads
    intermediate = config.intermediate_size or (hidden * 4)

    activations_bytes = calc_memory_activations(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        hidden_size=hidden,
        num_layers=layers,
        num_attention_heads=heads,
        use_recomputation=config.use_recomputation,
        tensor_parallel=tp,
    )

    # Model states are divided by TP and PP
    model_states_per_gpu = model_states_bytes / (tp * pp)

    # Activations are divided by PP (each stage only stores its layers' activations)
    # Note: TP division is already handled inside calc_memory_activations
    activations_per_gpu = activations_bytes / pp

    total_memory_bytes = model_states_per_gpu + activations_per_gpu
    memory_gb = total_memory_bytes / 1e9

    # Check if memory fits
    if memory_gb > hardware.memory_gb * 0.9:  # 90% threshold
        warnings.append(
            f"Memory usage ({memory_gb:.1f} GB) may exceed GPU capacity ({hardware.memory_gb} GB)"
        )

    memory = MemoryBreakdown(
        model_states=model_states_per_gpu / 1e9,  # Per-GPU value for consistency
        activations=activations_per_gpu / 1e9,    # Per-GPU value for consistency
        kv_cache=0,  # KV cache is for inference
        total=(model_states_bytes + activations_bytes) / 1e9,  # Total before parallelism
        per_gpu=memory_gb,
    )

    # Calculate training time
    training_seconds = calc_training_time(
        total_flops=total_flops,
        num_gpus=config.num_gpus,
        peak_tflops=hardware.peak_tflops_bf16,
        mfu=mfu,
    )

    training_hours = training_seconds / 3600
    training_days = training_hours / 24
    gpu_hours = training_hours * config.num_gpus
    gpu_days = gpu_hours / 24

    # Add notes
    if config.use_recomputation:
        notes.append("Using activation checkpointing to reduce memory")
    if zero_stage != ZeROStage.ZERO_0:
        notes.append(f"Using ZeRO Stage {zero_stage.value} optimization")

    # Update parallelism config with actual values
    final_parallelism = ParallelismConfig(
        tensor_parallel=tp,
        pipeline_parallel=pp,
        data_parallel=dp,
        zero_stage=zero_stage,
        reason=parallelism.reason,
    )

    return ComputeResult(
        model_params_billion=config.params_billion,
        training_tokens_billion=config.tokens_billion,
        num_gpus=config.num_gpus,
        gpu_name=config.gpu_name,
        total_flops=total_flops,
        total_flops_formatted=format_flops(total_flops),
        memory=memory,
        training_time_hours=training_hours,
        training_time_days=training_days,
        training_time_formatted=format_time(training_seconds),
        parallelism=final_parallelism,
        mfu=mfu,
        gpu_hours=gpu_hours,
        gpu_days=gpu_days,
        warnings=warnings,
        notes=notes,
    )
