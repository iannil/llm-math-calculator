# LLM Math Calculator (LMC)

[English](README.md) | 中文

**AI 基础设施资源规划工具** — 估算大模型训练和推理所需的计算资源

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LMC 是一个面向大模型训练的"科学计算器"，将复杂的资源估算公式封装为极简接口。它能帮助你快速回答：

- 训练一个 70B 模型需要多少张 GPU？
- 用 64 张 A100 训练需要多长时间？
- 显存够不够？该用什么并行策略？

## 功能特性

- **核心计算引擎**: 基于 Megatron-LM 论文的数学模型，支持 FLOPs、显存、训练时间估算
- **并行策略推荐**: 自动推荐 TP/PP/DP/ZeRO 配置
- **硬件数据库**: 预置 A100、H100、H800、昇腾 910B 等主流 GPU 参数
- **模型预设库**: 支持 Llama-3、Mixtral、GPT-3、Qwen、DeepSeek 等模型
- **多种接口**: CLI 命令行 + Python API + Web UI

## 安装

```bash
# 基础安装
pip install lmc

# 包含 Web UI
pip install lmc[web]

# 开发环境
pip install lmc[dev]
```

从源码安装：

```bash
git clone https://github.com/iannil/llm-math-calculator.git
cd llm-math-calculator
pip install -e ".[dev,web]"
```

## 快速开始

### CLI 命令行

```bash
# 基础估算
lmc train --gpu A100-80G-SXM --params 70B --tokens 400B --num-gpus 64

# 指定目标训练天数，自动计算 GPU 数量
lmc train --gpu A100-80G-SXM --params 70B --tokens 400B --days 30

# 使用模型预设
lmc train --preset Llama-3-70B --tokens 400B --gpu H100-80G-SXM --num-gpus 128

# 交互式模式
lmc train -i

# 查看硬件列表
lmc hardware list

# 查看模型预设
lmc model list

# CI/CD 可行性检查
lmc check --params 70B --gpu A100-80G-SXM --num-gpus 64
```

输出示例：

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

# 完整资源估算
config = TrainingConfig(
    params_billion=70,
    tokens_billion=400,
    gpu_name="A100-80G-SXM",
    num_gpus=64,
    zero_stage=ZeROStage.ZERO_2,
    use_recomputation=True,
)
result = estimate_resources(config)

print(f"训练时间: {result.training_time_formatted}")
print(f"GPU Hours: {result.gpu_hours:,.0f}")
print(f"显存/GPU: {result.memory.per_gpu:.1f} GB")
print(f"推荐策略: TP={result.parallelism.tensor_parallel}, PP={result.parallelism.pipeline_parallel}")

# 单独计算
flops = calc_total_flops(params=70e9, tokens=400e9)
memory = calc_memory_model_states(params=70e9, zero_stage=ZeROStage.ZERO_2, num_gpus=8)

# 使用预设
preset = get_preset("Llama-3-70B")
print(f"Hidden size: {preset.hidden_size}, Layers: {preset.num_layers}")
```

### Web UI

```bash
# 启动 Web 界面
streamlit run src/lmc/web.py
# 或
lmc-web
```

## 核心公式

基于 [Megatron-LM](https://arxiv.org/abs/1909.08053) 论文和业界实践：

| 指标 | 公式 | 说明 |
| ------ | ------ | ------ |
| 训练算力 | `6 × P × D` | P=参数量, D=训练数据量 |
| 显存 (ZeRO-0) | `16 Bytes × P` | 参数+梯度+优化器状态 |
| 激活值 | `s × b × h × L × (34 + 5ah/s)` | s=序列长度, b=batch, h=hidden, L=层数 |
| KV Cache | `2 × b × s × h × L × 2` | 推理场景 |
| 训练时间 | `FLOPs / (GPUs × TFLOPS × MFU)` | MFU: 模型利用率 |

### ZeRO 优化

| 阶段 | 分区内容 | 显存节省 |
| ------ | ---------- | ---------- |
| ZeRO-0 | 无 | 基准 (16B/param) |
| ZeRO-1 | 优化器状态 | ~4x |
| ZeRO-2 | 优化器+梯度 | ~8x |
| ZeRO-3 | 全部 | ~N (GPU数) |

## 支持的硬件

| 硬件 | 显存 | FP16 TFLOPS | 互联带宽 | 典型 MFU |
| ------ | ------ | ------------- | ---------- | ---------- |
| A100-40G-SXM | 40 GB | 312 | NVLink 600 GB/s | 50% |
| A100-80G-SXM | 80 GB | 312 | NVLink 600 GB/s | 50% |
| A100-80G-PCIe | 80 GB | 312 | - | 45% |
| H100-80G-SXM | 80 GB | 989 | NVLink 900 GB/s | 55% |
| H100-80G-PCIe | 80 GB | 756 | - | 50% |
| H800-80G-SXM | 80 GB | 989 | NVLink 400 GB/s | 50% |
| Ascend 910B | 64 GB | 320 | HCCS 392 GB/s | 45% |
| L40S | 48 GB | 362 | - | 45% |

## 支持的模型预设

| 模型 | 参数量 | 架构 | MoE |
| ------ | -------- | ------ | ----- |
| Llama-3-8B | 8B | Llama | - |
| Llama-3-70B | 70B | Llama | - |
| Llama-3.1-405B | 405B | Llama | - |
| Mixtral-8x7B | 46.7B (12.9B active) | Mixtral | 8×2 |
| Mixtral-8x22B | 141B (39B active) | Mixtral | 8×2 |
| GPT-3-175B | 175B | GPT | - |
| Qwen2-72B | 72B | Qwen | - |
| DeepSeek-V2-236B | 236B (21B active) | DeepSeek | 160×6 |

## 项目结构

```
llm-math-calculator/
├── pyproject.toml          # 项目配置
├── data/
│   ├── hardware.json       # 硬件数据库
│   └── presets.json        # 模型预设库
├── src/lmc/
│   ├── __init__.py         # API 导出
│   ├── cli.py              # CLI 命令
│   ├── engine.py           # 计算引擎
│   ├── loader.py           # 数据加载器
│   ├── models.py           # 数据模型
│   ├── optimizer.py        # 策略推荐器
│   └── web.py              # Web UI
└── tests/
    └── test_engine.py      # 单元测试
```

## 开发

```bash
# 克隆仓库
git clone https://github.com/iannil/llm-math-calculator.git
cd llm-math-calculator

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装开发依赖
pip install -e ".[dev,web]"

# 运行测试
pytest tests/ -v

# 运行单个命令测试
lmc train --gpu A100-80G-SXM --params 7B --tokens 100B
```

## 贡献

欢迎贡献！特别是：

- **硬件数据**: 添加新 GPU (如 B200、MI300X) 到 `data/hardware.json`
- **模型预设**: 添加新模型到 `data/presets.json`
- **公式优化**: 改进估算精度
- **文档翻译**: 多语言支持

提交 PR 前请确保：

1. 通过所有测试 (`pytest tests/`)
2. 代码符合项目风格
3. 更新相关文档

## 参考文献

- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

## License

MIT License - 详见 [LICENSE](LICENSE) 文件
