"""Unit tests for the LMC computation engine."""

import pytest

from lmc import (
    ZeROStage,
    calc_memory_activations,
    calc_memory_kv_cache,
    calc_memory_model_states,
    calc_total_flops,
    calc_training_time,
    format_bytes,
    format_flops,
    format_time,
    get_hardware,
    get_preset,
    list_hardware_names,
    list_preset_names,
)


class TestCalcTotalFlops:
    """Tests for calc_total_flops."""

    def test_basic_calculation(self):
        """Test basic FLOPs calculation: 6 * P * D."""
        params = 7_000_000_000  # 7B
        tokens = 100_000_000_000  # 100B
        flops = calc_total_flops(params, tokens)
        expected = 6 * params * tokens
        assert flops == expected

    def test_70b_model(self):
        """Test FLOPs for 70B model with 400B tokens."""
        params = 70_000_000_000
        tokens = 400_000_000_000
        flops = calc_total_flops(params, tokens)
        assert flops == 6 * 70e9 * 400e9
        assert flops == 1.68e23  # 168 ZFLOPs

    def test_moe_uses_active_params(self):
        """MoE models should use active_params for FLOPs."""
        total_params = 46_700_000_000  # Mixtral total
        active_params = 12_900_000_000  # Mixtral active
        tokens = 100_000_000_000

        flops = calc_total_flops(active_params, tokens, is_moe=True)
        expected = 6 * active_params * tokens
        assert flops == expected


class TestCalcMemoryModelStates:
    """Tests for calc_memory_model_states."""

    def test_zero_0_fp16(self):
        """ZeRO-0 with fp16: 16 bytes per parameter."""
        params = 1_000_000_000  # 1B
        memory = calc_memory_model_states(params, ZeROStage.ZERO_0)
        assert memory == 16 * params  # 16 GB

    def test_zero_1_partitions_optimizer(self):
        """ZeRO-1 partitions optimizer states across GPUs."""
        params = 1_000_000_000
        num_gpus = 8
        memory = calc_memory_model_states(params, ZeROStage.ZERO_1, num_gpus=num_gpus)
        # Optimizer (8 bytes) partitioned, rest (8 bytes) not
        expected = params * 8 / num_gpus + params * 8
        assert memory == expected

    def test_zero_2_partitions_optimizer_and_grads(self):
        """ZeRO-2 partitions optimizer states and gradients."""
        params = 1_000_000_000
        num_gpus = 8
        memory = calc_memory_model_states(params, ZeROStage.ZERO_2, num_gpus=num_gpus)
        # Optimizer + grads (10 bytes) partitioned, params (6 bytes) not
        expected = params * 10 / num_gpus + params * 6
        assert memory == expected

    def test_zero_3_partitions_everything(self):
        """ZeRO-3 partitions all model states."""
        params = 1_000_000_000
        num_gpus = 8
        memory = calc_memory_model_states(params, ZeROStage.ZERO_3, num_gpus=num_gpus)
        expected = 16 * params / num_gpus
        assert memory == expected


class TestCalcMemoryActivations:
    """Tests for calc_memory_activations."""

    def test_basic_activation_memory(self):
        """Test activation memory calculation."""
        memory = calc_memory_activations(
            batch_size=1,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
        )
        assert memory > 0

    def test_recomputation_reduces_memory(self):
        """Activation checkpointing should reduce memory."""
        base_memory = calc_memory_activations(
            batch_size=1,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            use_recomputation=False,
        )
        recompute_memory = calc_memory_activations(
            batch_size=1,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            use_recomputation=True,
        )
        assert recompute_memory < base_memory

    def test_tensor_parallel_reduces_memory(self):
        """Tensor parallelism should reduce activation memory."""
        base_memory = calc_memory_activations(
            batch_size=1,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            tensor_parallel=1,
        )
        tp_memory = calc_memory_activations(
            batch_size=1,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            tensor_parallel=4,
        )
        assert tp_memory == base_memory / 4


class TestCalcMemoryKvCache:
    """Tests for calc_memory_kv_cache."""

    def test_basic_kv_cache(self):
        """Test KV cache calculation: 2 * batch * seq * hidden * layers * 2 bytes."""
        memory = calc_memory_kv_cache(
            batch_size=1,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
        )
        expected = 2 * 1 * 4096 * 4096 * 32 * 2
        assert memory == expected

    def test_gqa_reduces_kv_cache(self):
        """GQA (fewer KV heads) should reduce KV cache."""
        mha_memory = calc_memory_kv_cache(
            batch_size=1,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            num_kv_heads=32,
            num_attention_heads=32,
        )
        gqa_memory = calc_memory_kv_cache(
            batch_size=1,
            seq_length=4096,
            hidden_size=4096,
            num_layers=32,
            num_kv_heads=8,
            num_attention_heads=32,
        )
        assert gqa_memory == mha_memory * (8 / 32)


class TestCalcTrainingTime:
    """Tests for calc_training_time."""

    def test_basic_training_time(self):
        """Test training time calculation."""
        total_flops = 1e21  # 1 ZFLOPs
        num_gpus = 8
        peak_tflops = 312  # A100
        mfu = 0.5

        time_seconds = calc_training_time(total_flops, num_gpus, peak_tflops, mfu)
        expected = total_flops / (num_gpus * peak_tflops * 1e12 * mfu)
        assert time_seconds == pytest.approx(expected)

    def test_more_gpus_faster(self):
        """More GPUs should reduce training time."""
        total_flops = 1e21
        time_8gpu = calc_training_time(total_flops, 8, 312, 0.5)
        time_16gpu = calc_training_time(total_flops, 16, 312, 0.5)
        assert time_16gpu == time_8gpu / 2

    def test_higher_mfu_faster(self):
        """Higher MFU should reduce training time."""
        total_flops = 1e21
        time_low_mfu = calc_training_time(total_flops, 8, 312, 0.4)
        time_high_mfu = calc_training_time(total_flops, 8, 312, 0.5)
        assert time_high_mfu < time_low_mfu


class TestFormatFunctions:
    """Tests for formatting functions."""

    def test_format_flops(self):
        """Test FLOPs formatting."""
        assert "TFLOP" in format_flops(1e12)
        assert "PFLOP" in format_flops(1e15)
        assert "EFLOP" in format_flops(1e18)
        assert "ZFLOP" in format_flops(1e21)

    def test_format_time(self):
        """Test time formatting."""
        assert "second" in format_time(30)
        assert "minute" in format_time(120)
        assert "hour" in format_time(7200)
        assert "day" in format_time(172800)

    def test_format_bytes(self):
        """Test bytes formatting."""
        assert "KB" in format_bytes(1e3)
        assert "MB" in format_bytes(1e6)
        assert "GB" in format_bytes(1e9)
        assert "TB" in format_bytes(1e12)


class TestDataLoaders:
    """Tests for data loading functions."""

    def test_load_hardware(self):
        """Test hardware loading."""
        names = list_hardware_names()
        assert len(names) > 0
        assert "A100-80G-SXM" in names
        assert "H100-80G-SXM" in names

    def test_get_hardware(self):
        """Test getting specific hardware."""
        hw = get_hardware("A100-80G-SXM")
        assert hw is not None
        assert hw.memory_gb == 80
        assert hw.peak_tflops_fp16 == 312

    def test_get_hardware_case_insensitive(self):
        """Test case-insensitive hardware lookup."""
        hw1 = get_hardware("A100-80G-SXM")
        hw2 = get_hardware("a100-80g-sxm")
        assert hw1 is not None
        assert hw2 is not None
        assert hw1.name == hw2.name

    def test_load_presets(self):
        """Test preset loading."""
        names = list_preset_names()
        assert len(names) > 0
        assert "Llama-3-70B" in names

    def test_get_preset(self):
        """Test getting specific preset."""
        preset = get_preset("Llama-3-70B")
        assert preset is not None
        assert preset.params_billion == 70
        assert preset.num_layers == 80

    def test_get_moe_preset(self):
        """Test getting MoE preset."""
        preset = get_preset("Mixtral-8x7B")
        assert preset is not None
        assert preset.is_moe
        assert preset.num_experts == 8
        assert preset.active_params_billion == 12.9
