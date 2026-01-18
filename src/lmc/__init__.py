"""LMC - LLM Math Calculator for AI infrastructure resource planning."""

from .engine import (
    calc_memory_activations,
    calc_memory_kv_cache,
    calc_memory_model_states,
    calc_total_flops,
    calc_training_time,
    estimate_resources,
    format_bytes,
    format_flops,
    format_time,
)
from .loader import (
    get_hardware,
    get_preset,
    list_hardware_names,
    list_preset_names,
    load_hardware,
    load_presets,
)
from .models import (
    ComputeResult,
    HardwareSpec,
    MemoryBreakdown,
    ModelPreset,
    ParallelismConfig,
    TrainingConfig,
    ZeROStage,
)
from .optimizer import (
    calc_required_gpus,
    check_network_bottleneck,
    estimate_communication_overhead,
    recommend_framework,
    recommend_parallelism,
    recommend_zero_stage,
)

__version__ = "0.1.0"

__all__ = [
    # Engine
    "calc_total_flops",
    "calc_memory_model_states",
    "calc_memory_activations",
    "calc_memory_kv_cache",
    "calc_training_time",
    "estimate_resources",
    "format_flops",
    "format_time",
    "format_bytes",
    # Loader
    "load_hardware",
    "load_presets",
    "get_hardware",
    "get_preset",
    "list_hardware_names",
    "list_preset_names",
    # Models
    "HardwareSpec",
    "ModelPreset",
    "TrainingConfig",
    "ComputeResult",
    "MemoryBreakdown",
    "ParallelismConfig",
    "ZeROStage",
    # Optimizer
    "recommend_parallelism",
    "recommend_zero_stage",
    "recommend_framework",
    "calc_required_gpus",
    "check_network_bottleneck",
    "estimate_communication_overhead",
]
