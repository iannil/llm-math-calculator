# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

LLM Math Calculator (LMC) 是一个 AI 基础设施资源规划工具。它用于估算大模型训练和推理所需的计算资源，包括 GPU 数量、显存占用和训练时间。

状态：设计阶段 - README.md 包含完整技术规格，但尚未实现代码。

## 项目指南

- 语言约定：交流与文档使用中文；生成的代码使用英文；文档放在 `docs` 且使用 Markdown。
- 发布约定：
  - 发布固定在 `/release` 文件夹，如 rust 服务固定发布在 `/release/rust` 文件夹。
  - 发布的成果物必须且始终以生产环境为标准，要包含所有发布生产所应该包含的文件或数据（包含全量发布与增量发布，首次发布与非首次发布）。
- 文档约定：
  - 每次修改都必须延续上一次的进展，每次修改的进展都必须保存在对应的 `docs` 文件夹下的文档中。
  - 执行修改过程中，进展随时保存文档，带上实际修改的时间，便于追溯修改历史。
  - 未完成的修改，文档保存在 `/docs/progress` 文件夹下。
  - 已完成的修改，文档保存在 `/docs/reports/completed` 文件夹下。
  - 对修改进行验收，文档保存在 `/docs/reports` 文件夹下。
  - 对重复的、冗余的、不能体现实际情况的文档或文档内容，要保持更新和调整。
  - 文档模板和命名规范可以参考 `/docs/standards` 和 `docs/templates` 文件夹下的内容。
- 数据约定：数据固定在`/data`文件夹下

### 面向大模型的可改写性（LLM Friendly）

- 一致的分层与目录：相同功能在各应用/包中遵循相同结构与命名，使检索与大范围重构更可控。
- 明确边界与单一职责：函数/类保持单一职责；公共模块暴露极少稳定接口；避免隐式全局状态。
- 显式类型与契约优先：导出 API 均有显式类型；运行时与编译时契约一致（zod schema 即类型源）。
- 声明式配置：将重要行为转为数据驱动（配置对象 + `as const`/`satisfies`），减少分支与条件散落。
- 可搜索性：统一命名（如 `parseXxx`、`assertNever`、`safeJsonParse`、`createXxxService`），降低 LLM 与人类的检索成本。
- 小步提交与计划：通过 `IMPLEMENTATION_PLAN.md` 和小步提交让模型理解上下文、意图与边界。
- 变更安全策略：批量程序性改动前先将原文件备份至 `/backup` 相对路径；若错误数异常上升，立即回滚备份。

## 规划架构

三种接口共享同一计算引擎：

- Python Library：核心算法库，可被其他系统集成
- CLI Tool：运维/开发人员的终端工具（`lmc train`、`lmc check`）
- Web UI：基于 Streamlit 或 React 的可视化界面

核心组件：

- 硬件数据库 (`hardware.json`)：GPU 参数（显存、算力、带宽），支持 A100、H100、昇腾 910B 等
- 模型预设库 (`presets.json`)：预配置的模型架构（Llama-3-70B 等）
- 计算引擎：基于 Megatron-LM 论文的数学模型
- 策略推荐器：并行策略建议（TP/PP/DP/ZeRO）

## 核心公式（来自 Megatron-LM）

- 训练算力：`6 * P * D`（P=参数量，D=训练数据量；MoE/SwiGLU 需修正）
- 显存占用 (ZeRO-0)：`16 Bytes * P`（参数 + 梯度 + 优化器状态）
- 训练时间：`Total FLOPs / (Num_GPUs * Peak_FLOPs * MFU)`
- KV Cache：`2 * batch * seq_len * hidden * layers * 2 Bytes`

MFU 默认值：A100/H100 约 0.45-0.55，昇腾 910B 约 0.4-0.5

## 实现注意事项

- MoE 模型需区分 `total_params`（总参数量）与 `active_params`（激活参数量）
- 需包含网络瓶颈估算（All-Reduce 通信开销）
- 支持重计算（激活检查点）的显存节省计算
