"""CLI interface for LMC."""

from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm
from rich.table import Table

from .engine import estimate_resources, format_bytes
from .loader import get_hardware, get_preset, list_hardware_names, list_preset_names, load_hardware, load_presets
from .models import TrainingConfig, ZeROStage
from .optimizer import check_network_bottleneck, recommend_framework, calc_required_gpus

app = typer.Typer(
    name="lmc",
    help="LLM Math Calculator - AI infrastructure resource planning tool",
    no_args_is_help=True,
)
console = Console()

hardware_app = typer.Typer(help="Hardware database commands")
model_app = typer.Typer(help="Model preset commands")
app.add_typer(hardware_app, name="hardware")
app.add_typer(model_app, name="model")


def parse_billion(value: str) -> float:
    """Parse a value with optional B suffix (e.g., '70B' -> 70.0)."""
    value = value.strip().upper()
    if value.endswith("B"):
        return float(value[:-1])
    return float(value)


def interactive_train():
    """Interactive mode for training estimation."""
    console.print("\n[bold cyan]LLM Math Calculator - Interactive Mode[/bold cyan]\n")

    # Step 1: Select GPU
    hardware_names = list_hardware_names()
    console.print("[dim]Available GPUs:[/dim]")
    for i, name in enumerate(hardware_names, 1):
        hw = get_hardware(name)
        console.print(f"  {i}. {name} ({hw.memory_gb}GB, {hw.peak_tflops_fp16} TFLOPS)")

    gpu_choice = Prompt.ask(
        "Select GPU",
        choices=[str(i) for i in range(1, len(hardware_names) + 1)] + hardware_names,
        default="1"
    )
    if gpu_choice.isdigit():
        gpu_name = hardware_names[int(gpu_choice) - 1]
    else:
        gpu_name = gpu_choice

    hw = get_hardware(gpu_name)
    console.print(f"[green]Selected: {gpu_name}[/green]\n")

    # Step 2: Model parameters or preset
    use_preset = Confirm.ask("Use a model preset?", default=False)

    if use_preset:
        preset_names = list_preset_names()
        console.print("[dim]Available presets:[/dim]")
        for i, name in enumerate(preset_names, 1):
            preset = get_preset(name)
            params_str = f"{preset.params_billion}B"
            if preset.active_params_billion:
                params_str += f" (active: {preset.active_params_billion}B)"
            console.print(f"  {i}. {name} - {params_str}")

        preset_choice = Prompt.ask(
            "Select preset",
            choices=[str(i) for i in range(1, len(preset_names) + 1)] + preset_names,
            default="1"
        )
        if preset_choice.isdigit():
            preset_name = preset_names[int(preset_choice) - 1]
        else:
            preset_name = preset_choice

        model_preset = get_preset(preset_name)
        params_b = model_preset.params_billion
        active_params_b = model_preset.active_params_billion
        hidden_size = model_preset.hidden_size
        num_layers = model_preset.num_layers
        num_heads = model_preset.num_attention_heads
        console.print(f"[green]Selected: {preset_name} ({params_b}B params)[/green]\n")
    else:
        params_input = Prompt.ask("Model parameters (e.g., 70B)", default="7B")
        params_b = parse_billion(params_input)
        active_params_b = None
        hidden_size = 4096
        num_layers = 32
        num_heads = 32

    # Step 3: Training data
    tokens_input = Prompt.ask("Training tokens (e.g., 400B)", default="100B")
    tokens_b = parse_billion(tokens_input)

    # Step 4: Target days or GPU count
    use_target_days = Confirm.ask("Specify target training days? (otherwise specify GPU count)", default=False)

    if use_target_days:
        target_days = FloatPrompt.ask("Target training days", default=30.0)
        mfu = hw.typical_mfu

        # Calculate required GPUs
        gpu_calc = calc_required_gpus(
            params_billion=params_b if active_params_b is None else active_params_b,
            tokens_billion=tokens_b,
            target_days=target_days,
            peak_tflops=hw.peak_tflops_bf16,
            mfu=mfu,
        )

        console.print(f"\n[cyan]GPU Calculation:[/cyan]")
        console.print(f"  Required GPUs (exact): {gpu_calc['required_gpus_exact']:.1f}")
        console.print(f"  Suggested GPUs: {gpu_calc['suggested_gpus']} ({gpu_calc['suggested_nodes']} nodes)")
        console.print(f"  Actual training time: {gpu_calc['actual_days']:.1f} days")

        num_gpus = gpu_calc['suggested_gpus']
        console.print(f"[green]Using {num_gpus} GPUs[/green]\n")
    else:
        num_gpus = IntPrompt.ask("Number of GPUs", default=8)

    # Step 5: Sequence length
    seq_length = IntPrompt.ask("Sequence length", default=4096)

    # Step 6: Batch size
    batch_size = IntPrompt.ask("Micro batch size", default=1)

    # Step 7: ZeRO stage
    zero_stage = IntPrompt.ask("ZeRO stage (0-3)", default=0)

    # Step 8: Recomputation
    use_recompute = Confirm.ask("Use activation checkpointing (recomputation)?", default=False)

    return {
        "gpu": gpu_name,
        "params_b": params_b,
        "active_params_b": active_params_b,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "tokens_b": tokens_b,
        "num_gpus": num_gpus,
        "seq_length": seq_length,
        "batch_size": batch_size,
        "zero_stage": zero_stage,
        "use_recompute": use_recompute,
    }


def print_full_report(result, hw, config, show_network=True):
    """Print the full training resource report."""
    console.print()

    # Header
    header_text = (
        f"[bold cyan]LLM Math Calculator Report[/bold cyan]\n"
        f"{'=' * 50}"
    )
    console.print(Panel(header_text, border_style="blue"))

    # Input section
    input_table = Table(title="[Input]", show_header=False, box=None, padding=(0, 2))
    input_table.add_column("Property", style="dim")
    input_table.add_column("Value", style="white")
    input_table.add_row("Model", f"{result.model_params_billion}B Params")
    input_table.add_row("Data", f"{result.training_tokens_billion}B Tokens")
    input_table.add_row("Hardware", f"{result.gpu_name} (FP16 Peak: {hw.peak_tflops_fp16} TFLOPS)")
    input_table.add_row("GPUs", f"{result.num_gpus}")
    console.print(input_table)
    console.print()

    # Resources Required section
    resources_table = Table(title="[Resources Required]", show_header=False, box=None, padding=(0, 2))
    resources_table.add_column("Property", style="dim")
    resources_table.add_column("Value", style="green")
    resources_table.add_row("Total FLOPs", result.total_flops_formatted)
    resources_table.add_row("Training Time", result.training_time_formatted)
    resources_table.add_row("GPU Hours", f"{result.gpu_hours:,.0f}")
    resources_table.add_row("GPU Days", f"{result.gpu_days:,.1f}")
    console.print(resources_table)
    console.print()

    # Memory Breakdown section
    memory_table = Table(title="[Memory Breakdown (per GPU)]", show_header=False, box=None, padding=(0, 2))
    memory_table.add_column("Property", style="dim")
    memory_table.add_column("Value", style="green")

    # Calculate component breakdown
    zero_stage = config.zero_stage
    if zero_stage == ZeROStage.ZERO_0:
        param_mem = result.model_params_billion * 2  # FP16 params
        grad_mem = result.model_params_billion * 2   # FP16 grads
        optim_mem = result.model_params_billion * 12  # Adam optimizer
    elif zero_stage == ZeROStage.ZERO_1:
        dp = result.parallelism.data_parallel
        param_mem = result.model_params_billion * 2
        grad_mem = result.model_params_billion * 2
        optim_mem = result.model_params_billion * 12 / dp
    elif zero_stage == ZeROStage.ZERO_2:
        dp = result.parallelism.data_parallel
        param_mem = result.model_params_billion * 2
        grad_mem = result.model_params_billion * 2 / dp
        optim_mem = result.model_params_billion * 12 / dp
    else:  # ZeRO-3
        dp = result.parallelism.data_parallel
        param_mem = result.model_params_billion * 2 / dp
        grad_mem = result.model_params_billion * 2 / dp
        optim_mem = result.model_params_billion * 12 / dp

    tp_pp = result.parallelism.tensor_parallel * result.parallelism.pipeline_parallel
    param_mem /= tp_pp
    grad_mem /= tp_pp
    optim_mem /= tp_pp

    memory_table.add_row("Parameters", f"{param_mem:.1f} GB")
    memory_table.add_row("Gradients", f"{grad_mem:.1f} GB")
    memory_table.add_row("Optimizer States", f"{optim_mem:.1f} GB")
    memory_table.add_row("Activations", f"{result.memory.activations:.1f} GB")

    # Buffer/fragmentation estimate (typically 5-10% overhead)
    buffer_mem = result.memory.per_gpu * 0.08
    memory_table.add_row("Buffer/Frag (~8%)", f"{buffer_mem:.1f} GB")

    total_with_buffer = result.memory.per_gpu + buffer_mem
    memory_table.add_row("-" * 20, "-" * 10)
    memory_table.add_row("Total Usage", f"{total_with_buffer:.1f} GB / {hw.memory_gb} GB")

    # Safety indicator
    usage_ratio = total_with_buffer / hw.memory_gb
    if usage_ratio < 0.8:
        safety = "[green](Safe)[/green]"
    elif usage_ratio < 0.95:
        safety = "[yellow](Tight)[/yellow]"
    else:
        safety = "[red](Risk OOM)[/red]"
    memory_table.add_row("Status", safety)

    console.print(memory_table)
    console.print()

    # Recommended Strategy section
    strategy_table = Table(title="[Recommended Strategy]", show_header=False, box=None, padding=(0, 2))
    strategy_table.add_column("Property", style="dim")
    strategy_table.add_column("Value", style="cyan")

    # Get framework recommendation
    framework_rec = recommend_framework(
        params_billion=result.model_params_billion,
        num_gpus=result.num_gpus,
        tensor_parallel=result.parallelism.tensor_parallel,
        pipeline_parallel=result.parallelism.pipeline_parallel,
        zero_stage=result.parallelism.zero_stage,
    )

    strategy_table.add_row("Framework", framework_rec["framework"])
    strategy_table.add_row(
        "Parallelism",
        f"TP={result.parallelism.tensor_parallel}, PP={result.parallelism.pipeline_parallel}, DP={result.parallelism.data_parallel}"
    )
    strategy_table.add_row("ZeRO Stage", f"Stage-{result.parallelism.zero_stage.value}")
    strategy_table.add_row(
        "FlashAttn",
        "[green]Enabled[/green] (Required)" if framework_rec["flash_attention"] else "Optional"
    )
    if framework_rec["gradient_checkpointing"]:
        strategy_table.add_row("Grad Checkpoint", "[green]Recommended[/green]")
    strategy_table.add_row("MFU", f"{result.mfu:.0%}")

    console.print(strategy_table)
    console.print()

    # Network analysis (if applicable)
    if show_network and result.parallelism.data_parallel > 1:
        interconnect_bw = hw.nvlink_bandwidth_gbps or hw.hccs_bandwidth_gbps or 100  # Default 100 GB/s
        network_analysis = check_network_bottleneck(
            params_billion=result.model_params_billion,
            num_gpus=result.num_gpus,
            tensor_parallel=result.parallelism.tensor_parallel,
            interconnect_bandwidth_gbps=interconnect_bw,
        )

        if network_analysis["is_network_bottleneck"]:
            console.print("[yellow]Network Warning:[/yellow]")
            console.print(f"  All-Reduce volume: {network_analysis['allreduce_volume_gb']:.1f} GB")
            console.print(f"  All-Reduce time: {network_analysis['allreduce_time_ms']:.0f} ms")
            console.print(f"  {network_analysis['recommendation']}")
            console.print()

    # Warnings and notes
    if result.warnings:
        for warning in result.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")

    if result.notes:
        for note in result.notes:
            console.print(f"[dim]Note:[/dim] {note}")

    # Footer
    console.print()
    console.print("=" * 50)
    console.print()


@hardware_app.command("list")
def hardware_list():
    """List all available hardware configurations."""
    table = Table(title="Available Hardware")
    table.add_column("Name", style="cyan")
    table.add_column("Vendor", style="green")
    table.add_column("Memory", justify="right")
    table.add_column("FP16 TFLOPS", justify="right")
    table.add_column("Bandwidth", justify="right")
    table.add_column("Typical MFU", justify="right")

    for hw in load_hardware():
        nvlink = f"{hw.nvlink_bandwidth_gbps} GB/s" if hw.nvlink_bandwidth_gbps else "N/A"
        table.add_row(
            hw.name,
            hw.vendor,
            f"{hw.memory_gb} GB",
            f"{hw.peak_tflops_fp16}",
            nvlink,
            f"{hw.typical_mfu:.0%}",
        )

    console.print(table)


@hardware_app.command("show")
def hardware_show(name: str):
    """Show details of a specific hardware."""
    hw = get_hardware(name)
    if hw is None:
        console.print(f"[red]Hardware '{name}' not found.[/red]")
        console.print(f"Available: {', '.join(list_hardware_names())}")
        raise typer.Exit(1)

    table = Table(title=f"Hardware: {hw.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Vendor", hw.vendor)
    table.add_row("Memory", f"{hw.memory_gb} GB")
    table.add_row("Memory Bandwidth", f"{hw.memory_bandwidth_gbps} GB/s")
    table.add_row("Peak FP16 TFLOPS", f"{hw.peak_tflops_fp16}")
    table.add_row("Peak BF16 TFLOPS", f"{hw.peak_tflops_bf16}")
    table.add_row("Peak FP32 TFLOPS", f"{hw.peak_tflops_fp32}")
    if hw.nvlink_bandwidth_gbps:
        table.add_row("NVLink Bandwidth", f"{hw.nvlink_bandwidth_gbps} GB/s")
    if hw.hccs_bandwidth_gbps:
        table.add_row("HCCS Bandwidth", f"{hw.hccs_bandwidth_gbps} GB/s")
    table.add_row("Typical MFU", f"{hw.typical_mfu:.0%}")
    table.add_row("TDP", f"{hw.tdp_watts} W")

    console.print(table)


@model_app.command("list")
def model_list():
    """List all available model presets."""
    table = Table(title="Available Model Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Architecture", style="green")
    table.add_column("Parameters", justify="right")
    table.add_column("Hidden Size", justify="right")
    table.add_column("Layers", justify="right")
    table.add_column("MoE", justify="center")

    for preset in load_presets():
        params_str = f"{preset.params_billion}B"
        if preset.active_params_billion:
            params_str += f" ({preset.active_params_billion}B active)"
        moe_str = f"{preset.num_experts}x{preset.num_experts_per_tok}" if preset.is_moe else "-"

        table.add_row(
            preset.name,
            preset.architecture,
            params_str,
            str(preset.hidden_size),
            str(preset.num_layers),
            moe_str,
        )

    console.print(table)


@model_app.command("show")
def model_show(name: str):
    """Show details of a specific model preset."""
    preset = get_preset(name)
    if preset is None:
        console.print(f"[red]Model preset '{name}' not found.[/red]")
        console.print(f"Available: {', '.join(list_preset_names())}")
        raise typer.Exit(1)

    table = Table(title=f"Model: {preset.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Architecture", preset.architecture)
    table.add_row("Total Parameters", f"{preset.params_billion}B")
    if preset.active_params_billion:
        table.add_row("Active Parameters", f"{preset.active_params_billion}B")
    table.add_row("Hidden Size", str(preset.hidden_size))
    table.add_row("Layers", str(preset.num_layers))
    table.add_row("Attention Heads", str(preset.num_attention_heads))
    table.add_row("KV Heads", str(preset.num_kv_heads))
    table.add_row("Intermediate Size", str(preset.intermediate_size))
    table.add_row("Vocab Size", str(preset.vocab_size))
    table.add_row("Max Sequence Length", str(preset.max_seq_length))
    table.add_row("Activation", preset.activation)
    if preset.is_moe:
        table.add_row("MoE Experts", f"{preset.num_experts}")
        table.add_row("Experts per Token", f"{preset.num_experts_per_tok}")

    console.print(table)


@app.command("train")
def train(
    gpu: Annotated[Optional[str], typer.Option("--gpu", "-g", help="GPU model name")] = None,
    params: Annotated[Optional[str], typer.Option("--params", "-p", help="Model parameters (e.g., 70B)")] = None,
    tokens: Annotated[Optional[str], typer.Option("--tokens", "-t", help="Training tokens (e.g., 400B)")] = None,
    num_gpus: Annotated[Optional[int], typer.Option("--num-gpus", "-n", help="Number of GPUs")] = None,
    days: Annotated[Optional[float], typer.Option("--days", "-d", help="Target training days (calculates GPU count)")] = None,
    preset: Annotated[Optional[str], typer.Option("--preset", help="Use a model preset")] = None,
    seq_length: Annotated[int, typer.Option("--seq-length", "-s", help="Sequence length")] = 4096,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Micro batch size")] = 1,
    mfu: Annotated[Optional[float], typer.Option("--mfu", help="Override MFU (0-1)")] = None,
    zero: Annotated[int, typer.Option("--zero", help="ZeRO stage (0-3)")] = 0,
    recompute: Annotated[bool, typer.Option("--recompute/--no-recompute", help="Use activation checkpointing")] = False,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Interactive mode")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON")] = False,
):
    """Estimate training resources for a model."""

    # Check if we should use interactive mode
    if interactive or (gpu is None and params is None and tokens is None):
        # Interactive mode
        try:
            config_dict = interactive_train()
            gpu = config_dict["gpu"]
            params_b = config_dict["params_b"]
            active_params_b = config_dict["active_params_b"]
            hidden_size = config_dict["hidden_size"]
            num_layers = config_dict["num_layers"]
            num_heads = config_dict["num_heads"]
            tokens_b = config_dict["tokens_b"]
            num_gpus = config_dict["num_gpus"]
            seq_length = config_dict["seq_length"]
            batch_size = config_dict["batch_size"]
            zero = config_dict["zero_stage"]
            recompute = config_dict["use_recompute"]
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)
    else:
        # Parameter mode - use defaults for unspecified values
        gpu = gpu or "A100-80G-SXM"
        params_b = parse_billion(params) if params else 7.0
        tokens_b = parse_billion(tokens) if tokens else 100.0

        # Use preset if specified
        active_params_b = None
        hidden_size = 4096
        num_layers = 32
        num_heads = 32

        if preset:
            model_preset = get_preset(preset)
            if model_preset is None:
                console.print(f"[red]Model preset '{preset}' not found.[/red]")
                raise typer.Exit(1)
            params_b = model_preset.params_billion
            active_params_b = model_preset.active_params_billion
            hidden_size = model_preset.hidden_size
            num_layers = model_preset.num_layers
            num_heads = model_preset.num_attention_heads
            seq_length = min(seq_length, model_preset.max_seq_length)

        # Handle --days parameter
        if days is not None:
            hw = get_hardware(gpu)
            if hw is None:
                console.print(f"[red]GPU '{gpu}' not found.[/red]")
                raise typer.Exit(1)

            effective_mfu = mfu or hw.typical_mfu
            gpu_calc = calc_required_gpus(
                params_billion=params_b if active_params_b is None else active_params_b,
                tokens_billion=tokens_b,
                target_days=days,
                peak_tflops=hw.peak_tflops_bf16,
                mfu=effective_mfu,
            )

            num_gpus = gpu_calc["suggested_gpus"]
            console.print(f"[cyan]Target: {days} days -> Suggested GPUs: {num_gpus} ({gpu_calc['suggested_nodes']} nodes)[/cyan]")
            console.print(f"[cyan]Actual training time: {gpu_calc['actual_days']:.1f} days[/cyan]\n")
        else:
            num_gpus = num_gpus or 8

    # Validate GPU
    hw = get_hardware(gpu)
    if hw is None:
        console.print(f"[red]GPU '{gpu}' not found.[/red]")
        console.print(f"Available: {', '.join(list_hardware_names())}")
        raise typer.Exit(1)

    # Create config
    config = TrainingConfig(
        params_billion=params_b,
        active_params_billion=active_params_b,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_heads,
        tokens_billion=tokens_b,
        batch_size=batch_size,
        seq_length=seq_length,
        gpu_name=gpu,
        num_gpus=num_gpus,
        mfu=mfu,
        zero_stage=ZeROStage(zero),
        use_recomputation=recompute,
    )

    # Calculate
    result = estimate_resources(config)

    if json_output:
        console.print(result.model_dump_json(indent=2))
        return

    # Print full report
    print_full_report(result, hw, config)


@app.command("check")
def check(
    params: Annotated[str, typer.Option("--params", "-p", help="Model parameters (e.g., 70B)")] = "7B",
    gpu: Annotated[str, typer.Option("--gpu", "-g", help="GPU model name")] = "A100-80G-SXM",
    num_gpus: Annotated[int, typer.Option("--num-gpus", "-n", help="Number of GPUs")] = 8,
    memory_threshold: Annotated[float, typer.Option("--memory-threshold", help="Memory threshold (0-1)")] = 0.9,
):
    """Check if a model configuration is feasible (for CI/CD)."""
    params_b = parse_billion(params)

    hw = get_hardware(gpu)
    if hw is None:
        console.print(f"[red]FAIL[/red]: GPU '{gpu}' not found")
        raise typer.Exit(1)

    config = TrainingConfig(
        params_billion=params_b,
        tokens_billion=1,  # Minimal for check
        gpu_name=gpu,
        num_gpus=num_gpus,
    )

    result = estimate_resources(config)

    if result.memory.per_gpu > hw.memory_gb * memory_threshold:
        console.print(
            f"[red]FAIL[/red]: Memory usage ({result.memory.per_gpu:.1f} GB) "
            f"exceeds threshold ({hw.memory_gb * memory_threshold:.1f} GB)"
        )
        raise typer.Exit(1)

    if result.warnings:
        for warning in result.warnings:
            console.print(f"[yellow]WARNING[/yellow]: {warning}")

    console.print(
        f"[green]PASS[/green]: {params_b}B model fits on {num_gpus}x {gpu} "
        f"({result.memory.per_gpu:.1f}/{hw.memory_gb} GB per GPU)"
    )


@app.command("calc-gpus")
def calc_gpus(
    params: Annotated[str, typer.Option("--params", "-p", help="Model parameters (e.g., 70B)")] = "7B",
    tokens: Annotated[str, typer.Option("--tokens", "-t", help="Training tokens (e.g., 400B)")] = "100B",
    days: Annotated[float, typer.Option("--days", "-d", help="Target training days")] = 30,
    gpu: Annotated[str, typer.Option("--gpu", "-g", help="GPU model name")] = "A100-80G-SXM",
    mfu: Annotated[Optional[float], typer.Option("--mfu", help="Override MFU (0-1)")] = None,
):
    """Calculate required GPUs for target training time."""
    params_b = parse_billion(params)
    tokens_b = parse_billion(tokens)

    hw = get_hardware(gpu)
    if hw is None:
        console.print(f"[red]GPU '{gpu}' not found.[/red]")
        console.print(f"Available: {', '.join(list_hardware_names())}")
        raise typer.Exit(1)

    effective_mfu = mfu or hw.typical_mfu

    result = calc_required_gpus(
        params_billion=params_b,
        tokens_billion=tokens_b,
        target_days=days,
        peak_tflops=hw.peak_tflops_bf16,
        mfu=effective_mfu,
    )

    table = Table(title="GPU Calculation")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", f"{params_b}B parameters")
    table.add_row("Data", f"{tokens_b}B tokens")
    table.add_row("Target Time", f"{days} days")
    table.add_row("GPU", f"{gpu} ({hw.peak_tflops_bf16} TFLOPS)")
    table.add_row("MFU", f"{effective_mfu:.0%}")
    table.add_row("-" * 20, "-" * 15)
    table.add_row("Required GPUs (exact)", f"{result['required_gpus_exact']:.1f}")
    table.add_row("Suggested GPUs", f"{result['suggested_gpus']}")
    table.add_row("Suggested Nodes (8 GPUs/node)", f"{result['suggested_nodes']}")
    table.add_row("Actual Training Time", f"{result['actual_days']:.1f} days")

    status = "[green]Meets target[/green]" if result['meets_target'] else "[yellow]Exceeds target[/yellow]"
    table.add_row("Status", status)

    console.print(table)


@app.callback()
def main():
    """LMC - LLM Math Calculator for AI infrastructure planning."""
    pass


if __name__ == "__main__":
    app()
