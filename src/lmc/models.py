"""Pydantic data models for LMC."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ZeROStage(int, Enum):
    """ZeRO optimization stages."""

    ZERO_0 = 0  # No ZeRO optimization
    ZERO_1 = 1  # Optimizer state partitioning
    ZERO_2 = 2  # Optimizer + gradient partitioning
    ZERO_3 = 3  # Optimizer + gradient + parameter partitioning


class ActivationType(str, Enum):
    """Activation function types."""

    GELU = "gelu"
    SILU = "silu"
    SWIGLU = "swiglu"


class HardwareSpec(BaseModel):
    """Hardware specification."""

    name: str
    vendor: str
    memory_gb: float
    memory_bandwidth_gbps: float
    peak_tflops_fp16: float
    peak_tflops_bf16: float
    peak_tflops_fp32: float
    nvlink_bandwidth_gbps: float = 0
    hccs_bandwidth_gbps: float = 0  # Huawei HCCS
    typical_mfu: float = Field(default=0.5, ge=0, le=1)
    tdp_watts: float = 0


class ModelPreset(BaseModel):
    """Model preset configuration."""

    name: str
    architecture: str
    params_billion: float
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_seq_length: int
    is_moe: bool = False
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    active_params_billion: Optional[float] = None
    activation: str = "silu"

    @property
    def params(self) -> int:
        """Total parameters in raw count."""
        return int(self.params_billion * 1e9)

    @property
    def active_params(self) -> int:
        """Active parameters for MoE models."""
        if self.active_params_billion is not None:
            return int(self.active_params_billion * 1e9)
        return self.params


class TrainingConfig(BaseModel):
    """Training configuration for resource estimation."""

    # Model parameters
    params_billion: float = Field(..., gt=0, description="Total model parameters in billions")
    active_params_billion: Optional[float] = Field(
        default=None, gt=0, description="Active parameters for MoE (billions)"
    )
    hidden_size: int = Field(default=4096, gt=0)
    num_layers: int = Field(default=32, gt=0)
    num_attention_heads: int = Field(default=32, gt=0)
    num_kv_heads: Optional[int] = Field(default=None, gt=0)
    intermediate_size: Optional[int] = Field(default=None, gt=0)

    # Training parameters
    tokens_billion: float = Field(..., gt=0, description="Training tokens in billions")
    batch_size: int = Field(default=1, gt=0, description="Micro batch size per GPU")
    seq_length: int = Field(default=4096, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)

    # Hardware
    gpu_name: str = Field(default="A100-80G-SXM")
    num_gpus: int = Field(default=8, gt=0)
    mfu: Optional[float] = Field(default=None, ge=0, le=1, description="Override MFU")

    # Optimization
    zero_stage: ZeROStage = Field(default=ZeROStage.ZERO_0)
    use_recomputation: bool = Field(default=False, description="Use activation checkpointing")
    use_fp16: bool = Field(default=True)

    # Parallelism (optional, for manual override)
    tensor_parallel: int = Field(default=1, ge=1)
    pipeline_parallel: int = Field(default=1, ge=1)
    data_parallel: Optional[int] = Field(default=None, ge=1)

    @property
    def params(self) -> int:
        """Total parameters in raw count."""
        return int(self.params_billion * 1e9)

    @property
    def active_params(self) -> int:
        """Active parameters for computation."""
        if self.active_params_billion is not None:
            return int(self.active_params_billion * 1e9)
        return self.params

    @property
    def tokens(self) -> int:
        """Training tokens in raw count."""
        return int(self.tokens_billion * 1e9)

    @property
    def effective_batch_size(self) -> int:
        """Global batch size across all GPUs and accumulation steps."""
        dp = self.data_parallel or (self.num_gpus // (self.tensor_parallel * self.pipeline_parallel))
        return self.batch_size * dp * self.gradient_accumulation_steps

    @property
    def kv_heads(self) -> int:
        """KV heads for GQA/MQA."""
        return self.num_kv_heads or self.num_attention_heads


class MemoryBreakdown(BaseModel):
    """Detailed memory usage breakdown in GB."""

    model_states: float = Field(description="Parameters + gradients + optimizer states")
    activations: float = Field(description="Activation memory per micro-batch")
    kv_cache: float = Field(default=0, description="KV cache for inference")
    total: float = Field(description="Total memory requirement")
    per_gpu: float = Field(description="Memory per GPU after parallelism")


class ParallelismConfig(BaseModel):
    """Recommended parallelism configuration."""

    tensor_parallel: int = Field(ge=1)
    pipeline_parallel: int = Field(ge=1)
    data_parallel: int = Field(ge=1)
    zero_stage: ZeROStage
    reason: str = Field(description="Explanation for the recommendation")


class ComputeResult(BaseModel):
    """Complete resource estimation result."""

    # Basic info
    model_params_billion: float
    training_tokens_billion: float
    num_gpus: int
    gpu_name: str

    # Compute
    total_flops: float = Field(description="Total FLOPs for training")
    total_flops_formatted: str = Field(description="Human-readable FLOPs")

    # Memory
    memory: MemoryBreakdown

    # Time
    training_time_hours: float
    training_time_days: float
    training_time_formatted: str

    # Parallelism
    parallelism: ParallelismConfig

    # Efficiency metrics
    mfu: float = Field(description="Model FLOPS Utilization")
    gpu_hours: float
    gpu_days: float

    # Warnings and notes
    warnings: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
