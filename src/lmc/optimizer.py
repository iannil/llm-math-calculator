"""Strategy optimizer for parallelism and ZeRO recommendations."""

from typing import Optional

from .models import ParallelismConfig, ZeROStage


def recommend_parallelism(
    params_billion: float,
    memory_gb: float,
    num_gpus: int,
    has_nvlink: bool = True,
    seq_length: int = 4096,
) -> ParallelismConfig:
    """
    Recommend parallelism strategy based on model size and hardware.

    General guidelines:
    - TP (Tensor Parallel): Reduces memory per GPU, requires fast interconnect
    - PP (Pipeline Parallel): Further memory reduction, increases latency
    - DP (Data Parallel): Scales throughput, no memory reduction per GPU
    - ZeRO: Memory optimization within data parallel group

    Args:
        params_billion: Model parameters in billions
        memory_gb: GPU memory in GB
        num_gpus: Total number of GPUs
        has_nvlink: Whether GPUs have NVLink/fast interconnect
        seq_length: Sequence length

    Returns:
        Recommended parallelism configuration
    """
    # Estimate memory requirement (rough: 16 bytes per param for fp16 training)
    min_memory_needed = params_billion * 16  # GB

    reasons = []

    # Determine TP based on model size and interconnect
    if params_billion >= 100:
        # Very large models (100B+): TP=8 if possible
        tp = min(8, num_gpus) if has_nvlink else min(4, num_gpus)
        reasons.append(f"Large model ({params_billion}B) needs TP={tp}")
    elif params_billion >= 30:
        # Large models (30-100B): TP=4-8
        tp = min(4, num_gpus) if has_nvlink else min(2, num_gpus)
        reasons.append(f"Medium-large model ({params_billion}B) uses TP={tp}")
    elif params_billion >= 10:
        # Medium models (10-30B): TP=2-4
        tp = min(2, num_gpus) if has_nvlink else 1
        reasons.append(f"Medium model ({params_billion}B) uses TP={tp}")
    else:
        # Small models (<10B): TP=1 usually sufficient
        tp = 1
        reasons.append(f"Small model ({params_billion}B) can use TP=1")

    # Determine PP based on remaining memory needs
    memory_after_tp = min_memory_needed / tp
    if memory_after_tp > memory_gb * 0.7:  # Need PP if still too large
        # Calculate needed PP
        pp_needed = int(memory_after_tp / (memory_gb * 0.5)) + 1
        pp = min(pp_needed, num_gpus // tp)
        pp = max(pp, 1)
        if pp > 1:
            reasons.append(f"Memory constraint requires PP={pp}")
    else:
        pp = 1

    # Calculate DP
    dp = num_gpus // (tp * pp)
    if dp < 1:
        # Adjust if we don't have enough GPUs
        dp = 1
        tp = min(tp, num_gpus)
        pp = min(pp, num_gpus // tp)

    # Determine ZeRO stage based on remaining memory pressure
    memory_per_gpu = min_memory_needed / (tp * pp)
    if memory_per_gpu > memory_gb * 0.8:
        zero_stage = ZeROStage.ZERO_3
        reasons.append("High memory pressure, recommend ZeRO-3")
    elif memory_per_gpu > memory_gb * 0.6:
        zero_stage = ZeROStage.ZERO_2
        reasons.append("Moderate memory pressure, recommend ZeRO-2")
    elif memory_per_gpu > memory_gb * 0.4 and dp > 1:
        zero_stage = ZeROStage.ZERO_1
        reasons.append("Using ZeRO-1 for optimizer memory savings")
    else:
        zero_stage = ZeROStage.ZERO_0
        reasons.append("Memory fits without ZeRO")

    return ParallelismConfig(
        tensor_parallel=tp,
        pipeline_parallel=pp,
        data_parallel=dp,
        zero_stage=zero_stage,
        reason="; ".join(reasons),
    )


def recommend_zero_stage(
    params_billion: float,
    memory_gb: float,
    num_gpus: int,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
) -> ZeROStage:
    """
    Recommend ZeRO stage based on memory requirements.

    Args:
        params_billion: Model parameters in billions
        memory_gb: GPU memory in GB
        num_gpus: Total number of GPUs
        tensor_parallel: TP degree
        pipeline_parallel: PP degree

    Returns:
        Recommended ZeRO stage
    """
    dp = num_gpus // (tensor_parallel * pipeline_parallel)
    memory_per_gpu = (params_billion * 16) / (tensor_parallel * pipeline_parallel)

    if memory_per_gpu > memory_gb * 0.8:
        return ZeROStage.ZERO_3
    elif memory_per_gpu > memory_gb * 0.6:
        return ZeROStage.ZERO_2
    elif memory_per_gpu > memory_gb * 0.4 and dp > 1:
        return ZeROStage.ZERO_1
    else:
        return ZeROStage.ZERO_0


def check_network_bottleneck(
    params_billion: float,
    num_gpus: int,
    tensor_parallel: int,
    interconnect_bandwidth_gbps: float,
    batch_size: int = 1,
    seq_length: int = 4096,
) -> dict:
    """
    Check if network is a bottleneck for distributed training.

    Args:
        params_billion: Model parameters in billions
        num_gpus: Total number of GPUs
        tensor_parallel: TP degree
        interconnect_bandwidth_gbps: Interconnect bandwidth in GB/s
        batch_size: Micro batch size
        seq_length: Sequence length

    Returns:
        Dictionary with bottleneck analysis
    """
    params = params_billion * 1e9

    # All-reduce communication volume per step (gradient sync)
    # Each parameter needs to be synchronized
    dp = num_gpus // tensor_parallel
    allreduce_volume_bytes = params * 4  # fp32 gradients

    # Ring all-reduce: 2 * (n-1) / n * data_size
    if dp > 1:
        ring_factor = 2 * (dp - 1) / dp
        actual_volume = allreduce_volume_bytes * ring_factor
    else:
        actual_volume = 0

    # Time for all-reduce (assuming full bandwidth utilization)
    bandwidth_bytes_per_sec = interconnect_bandwidth_gbps * 1e9
    allreduce_time_sec = actual_volume / bandwidth_bytes_per_sec if bandwidth_bytes_per_sec > 0 else 0

    # Tensor parallel communication (more frequent, smaller)
    # Approximate: 2 all-reduces per layer for TP
    tp_volume_per_layer = 2 * batch_size * seq_length * (params / 100) * 2  # rough estimate
    tp_time_sec = tp_volume_per_layer / bandwidth_bytes_per_sec if bandwidth_bytes_per_sec > 0 else 0

    is_bottleneck = allreduce_time_sec > 0.1  # More than 100ms is concerning

    return {
        "allreduce_volume_gb": actual_volume / 1e9,
        "allreduce_time_ms": allreduce_time_sec * 1000,
        "is_network_bottleneck": is_bottleneck,
        "recommendation": (
            "Network may be a bottleneck. Consider reducing DP or using gradient compression."
            if is_bottleneck
            else "Network bandwidth is sufficient."
        ),
    }


def recommend_framework(
    params_billion: float,
    num_gpus: int,
    tensor_parallel: int,
    pipeline_parallel: int,
    zero_stage: ZeROStage,
) -> dict:
    """
    Recommend training framework based on configuration.

    Args:
        params_billion: Model parameters in billions
        num_gpus: Total number of GPUs
        tensor_parallel: TP degree
        pipeline_parallel: PP degree
        zero_stage: ZeRO stage

    Returns:
        Dictionary with framework recommendations
    """
    recommendations = {
        "framework": "",
        "reason": "",
        "flash_attention": True,
        "flash_attention_reason": "Required for memory efficiency with long sequences",
        "gradient_checkpointing": False,
        "gradient_checkpointing_reason": "",
    }

    # Framework recommendation logic
    if tensor_parallel > 1 or pipeline_parallel > 1:
        # Need model parallelism
        if params_billion >= 100:
            recommendations["framework"] = "Megatron-LM"
            recommendations["reason"] = "Best for very large models with 3D parallelism"
        else:
            recommendations["framework"] = "Megatron-DeepSpeed"
            recommendations["reason"] = "Combines Megatron's parallelism with DeepSpeed's ZeRO"
    elif zero_stage >= ZeROStage.ZERO_2:
        recommendations["framework"] = "DeepSpeed"
        recommendations["reason"] = f"Optimized for ZeRO-{zero_stage.value} training"
    else:
        recommendations["framework"] = "PyTorch FSDP / DeepSpeed"
        recommendations["reason"] = "Standard distributed training sufficient"

    # Gradient checkpointing recommendation
    if params_billion >= 30:
        recommendations["gradient_checkpointing"] = True
        recommendations["gradient_checkpointing_reason"] = "Recommended for models > 30B to reduce activation memory"

    return recommendations


def calc_required_gpus(
    params_billion: float,
    tokens_billion: float,
    target_days: float,
    peak_tflops: float,
    mfu: float,
) -> dict:
    """
    Calculate required number of GPUs to finish training in target days.

    Args:
        params_billion: Model parameters in billions
        tokens_billion: Training tokens in billions
        target_days: Target training time in days
        peak_tflops: Peak TFLOPS per GPU
        mfu: Model FLOPS Utilization

    Returns:
        Dictionary with GPU requirements
    """
    params = params_billion * 1e9
    tokens = tokens_billion * 1e9
    total_flops = 6 * params * tokens

    # Target time in seconds
    target_seconds = target_days * 24 * 3600

    # Required compute power
    required_flops_per_sec = total_flops / target_seconds

    # Effective FLOPS per GPU
    effective_flops_per_gpu = peak_tflops * 1e12 * mfu

    # Required GPUs (raw calculation)
    required_gpus_raw = required_flops_per_sec / effective_flops_per_gpu

    # Round up to practical numbers (multiples of 8 for typical node sizes)
    import math
    if required_gpus_raw <= 8:
        suggested_gpus = 8
    elif required_gpus_raw <= 16:
        suggested_gpus = 16
    else:
        # Round up to next multiple of 8
        suggested_gpus = math.ceil(required_gpus_raw / 8) * 8

    # Calculate actual training time with suggested GPUs
    actual_seconds = total_flops / (suggested_gpus * effective_flops_per_gpu)
    actual_days = actual_seconds / (24 * 3600)

    return {
        "required_gpus_exact": required_gpus_raw,
        "suggested_gpus": suggested_gpus,
        "suggested_nodes": suggested_gpus // 8,
        "actual_days": actual_days,
        "meets_target": actual_days <= target_days,
    }


def estimate_communication_overhead(
    params_billion: float,
    num_gpus: int,
    tensor_parallel: int,
    pipeline_parallel: int,
    interconnect_bandwidth_gbps: float,
) -> float:
    """
    Estimate communication overhead as a fraction of compute time.

    Args:
        params_billion: Model parameters in billions
        num_gpus: Total number of GPUs
        tensor_parallel: TP degree
        pipeline_parallel: PP degree
        interconnect_bandwidth_gbps: Interconnect bandwidth in GB/s

    Returns:
        Estimated overhead fraction (0-1)
    """
    dp = num_gpus // (tensor_parallel * pipeline_parallel)

    # Base overhead from TP (within node, fast)
    tp_overhead = 0.02 * (tensor_parallel - 1) if tensor_parallel > 1 else 0

    # PP overhead (pipeline bubbles)
    # Bubble ratio ≈ (pp - 1) / (micro_batches + pp - 1)
    # Assuming reasonable micro-batch count
    assumed_micro_batches = 8
    pp_overhead = (pipeline_parallel - 1) / (assumed_micro_batches + pipeline_parallel - 1) if pipeline_parallel > 1 else 0

    # DP overhead (gradient sync)
    # Depends on interconnect speed
    if dp > 1 and interconnect_bandwidth_gbps > 0:
        # Rough estimate: larger models have more overlap opportunity
        dp_overhead = 0.05 * (dp - 1) / dp * (100 / max(interconnect_bandwidth_gbps, 100))
    else:
        dp_overhead = 0

    total_overhead = min(tp_overhead + pp_overhead + dp_overhead, 0.5)  # Cap at 50%
    return total_overhead
