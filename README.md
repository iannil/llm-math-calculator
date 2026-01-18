# LLM Math Calculator (LMC)

English | [中文](README_ZH.md)

**AI Infrastructure Resource Planning Tool** — Estimate computing resources required for LLM training and inference

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LMC is a "scientific calculator" for LLM training, encapsulating complex resource estimation formulas into a minimal interface. It helps you quickly answer:

- How many GPUs are needed to train a 70B model?
- How long will training take with 64 A100s?
- Is there enough memory? What parallelism strategy should be used?

## Features

- **Core Computation Engine**: Mathematical models based on Megatron-LM papers, supporting FLOPs, memory, and training time estimation
- **Parallelism Strategy Recommendations**: Automatic TP/PP/DP/ZeRO configuration suggestions
- **Hardware Database**: Pre-configured parameters for mainstream GPUs including A100, H100, H800, Ascend 910B
- **Model Presets**: Support for Llama-3, Mixtral, GPT-3, Qwen, DeepSeek and more
- **Multiple Interfaces**: CLI + Python API + Web UI

## Installation

```bash
# Basic installation
pip install lmc

# With Web UI
pip install lmc[web]

# Development environment
pip install lmc[dev]
```

Install from source:

```bash
git clone https://github.com/iannil/llm-math-calculator.git
cd llm-math-calculator
pip install -e ".[dev,web]"
```

## Quick Start

### CLI

```bash
# Basic estimation
lmc train --gpu A100-80G-SXM --params 70B --tokens 400B --num-gpus 64

# Specify target training days, auto-calculate GPU count
lmc train --gpu A100-80G-SXM --params 70B --tokens 400B --days 30

# Use model preset
lmc train --preset Llama-3-70B --tokens 400B --gpu H100-80G-SXM --num-gpus 128

# Interactive mode
lmc train -i

# List hardware
lmc hardware list

# List model presets
lmc model list

# CI/CD feasibility check
lmc check --params 70B --gpu A100-80G-SXM --num-gpus 64
```

Output example:

```
╭──────────────────────────────────────────────────────────────╮
│ LLM Math Calculator Report                                   │
╰──────────────────────────────────────────────────────────────╯
                       [Input]
  Model       70.0B Params
  Data        400.0B Tokens
  Hardware    A100-80G-SXM (FP16 Peak: 312.0 TFLOPS)
  GPUs        64

       [Resources Required]
  Total FLOPs      168.00 ZFLOPs
  Training Time    6.5 months
  GPU Hours        299,145
  GPU Days         12,464.4

        [Memory Breakdown (per GPU)]
  Parameters              4.4 GB
  Gradients               2.2 GB
  Optimizer States        13.1 GB
  Activations             9.2 GB
  Buffer/Frag (~8%)       2.7 GB
  --------------------    ----------
  Total Usage             35.9 GB / 80.0 GB
  Status                  (Safe)

         [Recommended Strategy]
  Framework          Megatron-DeepSpeed
  Parallelism        TP=4, PP=8, DP=2
  ZeRO Stage         Stage-2
  FlashAttn          Enabled (Required)
  Grad Checkpoint    Recommended
  MFU                50%
```

### Python API

```python
from lmc import (
    estimate_resources,
    TrainingConfig,
    ZeROStage,
    calc_total_flops,
    calc_memory_model_states,
    get_hardware,
    get_preset,
)

# Full resource estimation
config = TrainingConfig(
    params_billion=70,
    tokens_billion=400,
    gpu_name="A100-80G-SXM",
    num_gpus=64,
    zero_stage=ZeROStage.ZERO_2,
    use_recomputation=True,
)
result = estimate_resources(config)

print(f"Training time: {result.training_time_formatted}")
print(f"GPU Hours: {result.gpu_hours:,.0f}")
print(f"Memory/GPU: {result.memory.per_gpu:.1f} GB")
print(f"Recommended strategy: TP={result.parallelism.tensor_parallel}, PP={result.parallelism.pipeline_parallel}")

# Individual calculations
flops = calc_total_flops(params=70e9, tokens=400e9)
memory = calc_memory_model_states(params=70e9, zero_stage=ZeROStage.ZERO_2, num_gpus=8)

# Using presets
preset = get_preset("Llama-3-70B")
print(f"Hidden size: {preset.hidden_size}, Layers: {preset.num_layers}")
```

### Web UI

```bash
# Launch Web interface
streamlit run src/lmc/web.py
# Or
lmc-web
```

## Core Formulas

Based on [Megatron-LM](https://arxiv.org/abs/1909.08053) paper and industry practices:

| Metric | Formula | Description |
| -------- | --------- | ------------- |
| Training FLOPs | `6 × P × D` | P=parameters, D=training data size |
| Memory (ZeRO-0) | `16 Bytes × P` | Parameters + Gradients + Optimizer states |
| Activations | `s × b × h × L × (34 + 5ah/s)` | s=sequence length, b=batch, h=hidden, L=layers |
| KV Cache | `2 × b × s × h × L × 2` | For inference |
| Training Time | `FLOPs / (GPUs × TFLOPS × MFU)` | MFU: Model FLOPs Utilization |

### ZeRO Optimization

| Stage | Partitioned Content | Memory Savings |
| ------- | --------------------- | ---------------- |
| ZeRO-0 | None | Baseline (16B/param) |
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | Optimizer + Gradients | ~8x |
| ZeRO-3 | All | ~N (GPU count) |

## Supported Hardware

| Hardware | Memory | FP16 TFLOPS | Interconnect Bandwidth | Typical MFU |
| ---------- | -------- | ------------- | ------------------------ | ------------- |
| A100-40G-SXM | 40 GB | 312 | NVLink 600 GB/s | 50% |
| A100-80G-SXM | 80 GB | 312 | NVLink 600 GB/s | 50% |
| A100-80G-PCIe | 80 GB | 312 | - | 45% |
| H100-80G-SXM | 80 GB | 989 | NVLink 900 GB/s | 55% |
| H100-80G-PCIe | 80 GB | 756 | - | 50% |
| H800-80G-SXM | 80 GB | 989 | NVLink 400 GB/s | 50% |
| Ascend 910B | 64 GB | 320 | HCCS 392 GB/s | 45% |
| L40S | 48 GB | 362 | - | 45% |

## Supported Model Presets

| Model | Parameters | Architecture | MoE |
| ------- | ------------ | -------------- | ----- |
| Llama-3-8B | 8B | Llama | - |
| Llama-3-70B | 70B | Llama | - |
| Llama-3.1-405B | 405B | Llama | - |
| Mixtral-8x7B | 46.7B (12.9B active) | Mixtral | 8×2 |
| Mixtral-8x22B | 141B (39B active) | Mixtral | 8×2 |
| GPT-3-175B | 175B | GPT | - |
| Qwen2-72B | 72B | Qwen | - |
| DeepSeek-V2-236B | 236B (21B active) | DeepSeek | 160×6 |

## Project Structure

```
llm-math-calculator/
├── pyproject.toml          # Project configuration
├── data/
│   ├── hardware.json       # Hardware database
│   └── presets.json        # Model presets
├── src/lmc/
│   ├── __init__.py         # API exports
│   ├── cli.py              # CLI commands
│   ├── engine.py           # Computation engine
│   ├── loader.py           # Data loader
│   ├── models.py           # Data models
│   ├── optimizer.py        # Strategy recommender
│   └── web.py              # Web UI
└── tests/
    └── test_engine.py      # Unit tests
```

## Development

```bash
# Clone repository
git clone https://github.com/iannil/llm-math-calculator.git
cd llm-math-calculator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install development dependencies
pip install -e ".[dev,web]"

# Run tests
pytest tests/ -v

# Test a single command
lmc train --gpu A100-80G-SXM --params 7B --tokens 100B
```

## Contributing

Contributions are welcome! Especially:

- **Hardware Data**: Add new GPUs (e.g., B200, MI300X) to `data/hardware.json`
- **Model Presets**: Add new models to `data/presets.json`
- **Formula Improvements**: Enhance estimation accuracy
- **Documentation**: Multi-language support

Before submitting a PR, please ensure:

1. All tests pass (`pytest tests/`)
2. Code follows project style
3. Related documentation is updated

## References

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

## License

MIT License - See [LICENSE](LICENSE) file for details
