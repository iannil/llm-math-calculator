"""Streamlit Web UI for LMC."""

import streamlit as st

from .engine import estimate_resources, format_flops, format_time
from .loader import get_hardware, get_preset, list_hardware_names, list_preset_names, load_hardware, load_presets
from .models import TrainingConfig, ZeROStage
from .optimizer import check_network_bottleneck, recommend_framework, calc_required_gpus


def run_app():
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="LLM Math Calculator",
        page_icon="🧮",
        layout="wide",
    )

    st.title("🧮 LLM Math Calculator")
    st.markdown("**AI Infrastructure Resource Planning Tool**")
    st.markdown("---")

    # Sidebar for inputs
    with st.sidebar:
        st.header("Configuration")

        # Hardware selection
        st.subheader("Hardware")
        hardware_names = list_hardware_names()
        gpu_name = st.selectbox(
            "GPU Model",
            hardware_names,
            index=hardware_names.index("A100-80G-SXM") if "A100-80G-SXM" in hardware_names else 0,
        )
        hw = get_hardware(gpu_name)

        st.caption(f"Memory: {hw.memory_gb} GB | TFLOPS: {hw.peak_tflops_fp16}")

        # Model selection
        st.subheader("Model")
        use_preset = st.checkbox("Use Model Preset", value=False)

        if use_preset:
            preset_names = list_preset_names()
            preset_name = st.selectbox("Model Preset", preset_names)
            preset = get_preset(preset_name)
            params_b = preset.params_billion
            active_params_b = preset.active_params_billion
            hidden_size = preset.hidden_size
            num_layers = preset.num_layers
            num_heads = preset.num_attention_heads
            max_seq = preset.max_seq_length
            st.caption(f"Parameters: {params_b}B" + (f" (Active: {active_params_b}B)" if active_params_b else ""))
        else:
            params_b = st.slider("Model Parameters (Billion)", 1.0, 500.0, 7.0, 0.1)
            active_params_b = None
            hidden_size = 4096
            num_layers = 32
            num_heads = 32
            max_seq = 131072

        # Training data
        st.subheader("Training Data")
        tokens_b = st.slider("Training Tokens (Billion)", 10.0, 2000.0, 100.0, 10.0)
        seq_length = st.slider("Sequence Length", 512, min(max_seq, 131072), 4096, 512)

        # GPU configuration
        st.subheader("GPU Configuration")
        config_mode = st.radio("Configuration Mode", ["Specify GPUs", "Specify Target Days"])

        if config_mode == "Specify GPUs":
            num_gpus = st.slider("Number of GPUs", 1, 1024, 8, 1)
            target_days = None
        else:
            target_days = st.slider("Target Training Days", 1, 180, 30, 1)
            mfu_for_calc = hw.typical_mfu
            gpu_calc = calc_required_gpus(
                params_billion=params_b if active_params_b is None else active_params_b,
                tokens_billion=tokens_b,
                target_days=target_days,
                peak_tflops=hw.peak_tflops_bf16,
                mfu=mfu_for_calc,
            )
            num_gpus = gpu_calc["suggested_gpus"]
            st.info(f"Suggested: {num_gpus} GPUs ({gpu_calc['suggested_nodes']} nodes)")

        # Optimization
        st.subheader("Optimization")
        zero_stage = st.selectbox("ZeRO Stage", [0, 1, 2, 3], index=0)
        use_recompute = st.checkbox("Activation Checkpointing", value=False)
        batch_size = st.number_input("Micro Batch Size", 1, 64, 1)

        # MFU override
        st.subheader("Advanced")
        mfu_override = st.checkbox("Override MFU", value=False)
        if mfu_override:
            mfu = st.slider("MFU", 0.1, 0.8, hw.typical_mfu, 0.01)
        else:
            mfu = None

    # Main content area
    col1, col2 = st.columns(2)

    # Create config and calculate
    config = TrainingConfig(
        params_billion=params_b,
        active_params_billion=active_params_b,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        tokens_billion=tokens_b,
        batch_size=batch_size,
        seq_length=seq_length,
        gpu_name=gpu_name,
        num_gpus=num_gpus,
        mfu=mfu,
        zero_stage=ZeROStage(zero_stage),
        use_recomputation=use_recompute,
    )

    result = estimate_resources(config)

    with col1:
        st.header("📊 Resource Estimation")

        # Compute metrics
        st.subheader("Compute")
        compute_col1, compute_col2 = st.columns(2)
        with compute_col1:
            st.metric("Total FLOPs", result.total_flops_formatted)
        with compute_col2:
            st.metric("MFU", f"{result.mfu:.0%}")

        # Time metrics
        st.subheader("Training Time")
        time_col1, time_col2, time_col3 = st.columns(3)
        with time_col1:
            st.metric("Duration", result.training_time_formatted)
        with time_col2:
            st.metric("GPU Hours", f"{result.gpu_hours:,.0f}")
        with time_col3:
            st.metric("GPU Days", f"{result.gpu_days:,.1f}")

        # Memory breakdown
        st.subheader("Memory (per GPU)")

        # Calculate detailed breakdown
        if zero_stage == 0:
            param_mem = result.model_params_billion * 2
            grad_mem = result.model_params_billion * 2
            optim_mem = result.model_params_billion * 12
        elif zero_stage == 1:
            dp = result.parallelism.data_parallel
            param_mem = result.model_params_billion * 2
            grad_mem = result.model_params_billion * 2
            optim_mem = result.model_params_billion * 12 / dp
        elif zero_stage == 2:
            dp = result.parallelism.data_parallel
            param_mem = result.model_params_billion * 2
            grad_mem = result.model_params_billion * 2 / dp
            optim_mem = result.model_params_billion * 12 / dp
        else:
            dp = result.parallelism.data_parallel
            param_mem = result.model_params_billion * 2 / dp
            grad_mem = result.model_params_billion * 2 / dp
            optim_mem = result.model_params_billion * 12 / dp

        tp_pp = result.parallelism.tensor_parallel * result.parallelism.pipeline_parallel
        param_mem /= tp_pp
        grad_mem /= tp_pp
        optim_mem /= tp_pp

        buffer_mem = result.memory.per_gpu * 0.08
        total_with_buffer = result.memory.per_gpu + buffer_mem

        # Memory bar chart
        import pandas as pd
        memory_data = pd.DataFrame({
            "Component": ["Parameters", "Gradients", "Optimizer", "Activations", "Buffer"],
            "GB": [param_mem, grad_mem, optim_mem, result.memory.activations, buffer_mem]
        })
        st.bar_chart(memory_data.set_index("Component"))

        # Memory status
        usage_ratio = total_with_buffer / hw.memory_gb
        if usage_ratio < 0.8:
            st.success(f"Memory: {total_with_buffer:.1f} / {hw.memory_gb} GB (Safe)")
        elif usage_ratio < 0.95:
            st.warning(f"Memory: {total_with_buffer:.1f} / {hw.memory_gb} GB (Tight)")
        else:
            st.error(f"Memory: {total_with_buffer:.1f} / {hw.memory_gb} GB (Risk OOM)")

    with col2:
        st.header("🎯 Recommendations")

        # Parallelism strategy
        st.subheader("Parallelism Strategy")
        para_col1, para_col2, para_col3 = st.columns(3)
        with para_col1:
            st.metric("Tensor Parallel", result.parallelism.tensor_parallel)
        with para_col2:
            st.metric("Pipeline Parallel", result.parallelism.pipeline_parallel)
        with para_col3:
            st.metric("Data Parallel", result.parallelism.data_parallel)

        st.caption(f"ZeRO Stage: {result.parallelism.zero_stage.value}")

        # Framework recommendation
        st.subheader("Framework")
        framework_rec = recommend_framework(
            params_billion=result.model_params_billion,
            num_gpus=result.num_gpus,
            tensor_parallel=result.parallelism.tensor_parallel,
            pipeline_parallel=result.parallelism.pipeline_parallel,
            zero_stage=result.parallelism.zero_stage,
        )

        st.info(f"**{framework_rec['framework']}**\n\n{framework_rec['reason']}")

        if framework_rec["flash_attention"]:
            st.success("✅ FlashAttention: Required")
        if framework_rec["gradient_checkpointing"]:
            st.success("✅ Gradient Checkpointing: Recommended")

        # Network analysis
        if result.parallelism.data_parallel > 1:
            st.subheader("Network Analysis")
            interconnect_bw = hw.nvlink_bandwidth_gbps or hw.hccs_bandwidth_gbps or 100
            network = check_network_bottleneck(
                params_billion=result.model_params_billion,
                num_gpus=result.num_gpus,
                tensor_parallel=result.parallelism.tensor_parallel,
                interconnect_bandwidth_gbps=interconnect_bw,
            )

            if network["is_network_bottleneck"]:
                st.warning(f"⚠️ Network may be a bottleneck\n\nAll-Reduce: {network['allreduce_volume_gb']:.1f} GB ({network['allreduce_time_ms']:.0f} ms)")
            else:
                st.success("✅ Network bandwidth sufficient")

        # Warnings
        if result.warnings:
            st.subheader("Warnings")
            for warning in result.warnings:
                st.warning(warning)

    # Report generation
    st.markdown("---")
    st.header("📋 Report")

    report_text = f"""# LLM Math Calculator Report

## Configuration
- **Model**: {result.model_params_billion}B parameters
- **Training Data**: {result.training_tokens_billion}B tokens
- **Hardware**: {result.num_gpus}x {result.gpu_name}
- **Sequence Length**: {seq_length}

## Resource Requirements
- **Total FLOPs**: {result.total_flops_formatted}
- **Training Time**: {result.training_time_formatted}
- **GPU Hours**: {result.gpu_hours:,.0f}
- **GPU Days**: {result.gpu_days:,.1f}

## Memory Breakdown (per GPU)
- **Model States**: {result.memory.model_states:.1f} GB
- **Activations**: {result.memory.activations:.1f} GB
- **Total**: {result.memory.per_gpu:.1f} GB / {hw.memory_gb} GB

## Recommended Strategy
- **Framework**: {framework_rec['framework']}
- **Parallelism**: TP={result.parallelism.tensor_parallel}, PP={result.parallelism.pipeline_parallel}, DP={result.parallelism.data_parallel}
- **ZeRO Stage**: {result.parallelism.zero_stage.value}
- **FlashAttention**: {'Enabled' if framework_rec['flash_attention'] else 'Optional'}
- **MFU**: {result.mfu:.0%}

---
*Generated by LLM Math Calculator*
"""

    st.code(report_text, language="markdown")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        st.download_button(
            "📥 Download Report (Markdown)",
            report_text,
            file_name="lmc_report.md",
            mime="text/markdown",
        )
    with col_btn2:
        # JSON export
        json_data = result.model_dump_json(indent=2)
        st.download_button(
            "📥 Download Report (JSON)",
            json_data,
            file_name="lmc_report.json",
            mime="application/json",
        )


def main():
    """Entry point for the web application."""
    run_app()


if __name__ == "__main__":
    main()
