# Speedrun-MuP

基于 modded-nanogpt 速跑架构的 Maximal Update Parameterization (MuP) 研究实现，专为高效 Transformer 缩放实验设计。

## 概述

本项目将 modded-nanogpt 的最新训练优化与 MuP 理论的原理性缩放定律相结合，在保持竞争性训练性能的同时，实现模型宽度间的零样本超参数迁移。

**核心特性：**
- 完整 modded-nanogpt 架构实现 (FlexAttention、FP8、U-net 跳跃连接、数值嵌入、Muon 优化器)
- 标准 MuP 实现，包含 InfShape 维度跟踪和坐标检查
- 8xH100 系统分布式训练支持
- 全面的验证工具和实验跟踪

## 项目结构

```
speedrun_mup/
├── core/                   
│   ├── model.py            # modded-nanogpt 架构
│   ├── mup.py              # MuP 缩放和维度跟踪
│   ├── optimizers.py       # Muon 和 DistAdam 优化器
│   └── utils.py            # 日志、指标和工具函数
├── scripts/                
│   ├── run_basic_speedrun.sh          # 基础速跑训练
│   ├── run_mup_width.sh              # 单组宽度缩放实验
│   ├── run_mup_width_group_scaling.sh # 多组宽度系统验证
│   └── data_process/                  # 数据下载和处理
└── train.py                # 开始训练！
```

## 快速开始

### 环境准备

```bash
# 克隆仓库
git clone <repository-url>
cd speedrun_mup

# 安装依赖
pip install -r requirements.txt

# 下载训练数据 (推荐 10B 数据集)
bash scripts/data_process/download_hf_data.sh
```

### 基础训练 (无 MuP)

```bash
# 标准速跑训练
bash scripts/run_basic_speedrun.sh 768 1750 1337
# 参数: 模型宽度 训练步数 随机种子
```

### MuP 缩放实验

```bash
# 单组宽度缩放 (从基准宽度迁移超参数到目标宽度)
bash scripts/run_mup_width.sh 1024 768 1000 1337
# 参数: 目标宽度 基准宽度 训练步数 随机种子

# 多组宽度系统验证
bash scripts/run_mup_width_group_scaling.sh 256 "512 768 1024 1536" 1000 1337
# 参数: 基准宽度 "目标宽度列表" 每个宽度的训练步数 随机种子
```

### 直接使用训练脚本

```bash
# 基础训练
python train.py --width 768 --iterations 1750

# MuP 训练
python train.py --mup --width 1024 --base-width 768 --coord-check

# 分布式训练 (8xH100)
torchrun --nproc_per_node=8 train.py --mup --width 1024 --base-width 768
```

## 核心概念解释

### MuP (Maximal Update Parameterization)

MuP 是一种参数化方法，可以实现：
- **零样本超参数迁移**：在小模型上调优的超参数直接适用于大模型
- **稳定的激活幅度**：不同宽度的模型激活值保持相似的数量级
- **理论支持的缩放**：基于数学理论而非经验试错

### Width Sweep (宽度扫描)

宽度扫描是验证 MuP 正确性的标准方法：
1. 使用基准宽度 (如 256) 训练参考模型
2. 将相同超参数应用到不同目标宽度 (512, 768, 1024...)
3. 通过坐标检查验证激活幅度的稳定性
4. 如果 MuP 实现正确，所有宽度应展现相似的训练动态

### Coordinate Checking (坐标检查)

坐标检查通过可视化不同宽度模型的激活统计来验证 MuP：
- **正确的 MuP**: 激活均值和方差在不同宽度间稳定
- **错误的实现**: 激活幅度随宽度剧烈变化

## 架构细节

### modded-nanogpt 兼容性

模型实现与 modded-nanogpt 完全兼容：
- FP8 自定义算子 (`nanogpt::mm`)
- FlexAttention 滑动窗口块掩码
- U-net 跳跃连接和学习标量权重
- 数值嵌入 012...012 模式
- 半截断 RoPE 与基频调优
- ReLU² 激活和 logit 软限制

### MuP 集成

MuP 实现遵循标准做法：
- 所有参数的 InfShape 维度跟踪
- MuP 感知的初始化和优化器
- 输出层缩放和坐标检查
- 支持宽度缩放实验

### 性能优化

包含所有 modded-nanogpt 优化：
- 核心预热和 torch.compile
- 文档对齐的分布式数据加载
- 动量预热的学习率调度
- 内存高效的梯度累积

## 监控和日志系统

### 智能监控

**SimpleLogger**: 与 modded-nanogpt 风格对齐的简洁日志系统
- 自动实验命名：`speedrun_20250831_143022_mup_w1024_base768`
- 结构化日志：控制台 + 文件 + W&B
- 硬件检测：H100(989T), B200(2500T), A100(312T) FLOPS

**TrainingMonitor**: 性能感知的高级监控
- 基础指标 (每步): 梯度范数、参数范数
- 昂贵指标 (按间隔): 谱范数、激活统计
- 智能调度：平衡监控详细度与训练速度

### 性能开销

- **轻量监控** (<1ms/step): 基础指标、W&B 日志
- **昂贵监控** (50-200ms/step): 谱范数、激活统计
- **建议配置**: 日常训练每 100 步，研究调试每 10 步

## 实验组织

### 日志结构

```
logs/speedrun_20250831_143022_mup_w1024_base768/
├── training.log      # 带时间戳的训练日志
├── config.json       # 实验配置备份
└── console.log       # 控制台输出备份
```

### W&B 集成

- **项目分组**: `speedrun-basic`, `speedrun-mup`, `speedrun-mup-group`
- **自动命名**: 基于时间戳和配置的描述性名称
- **配置跟踪**: 完整的超参数记录
- **可视化**: 训练曲线、激活统计、坐标检查图

## 硬件要求

### 推荐配置
- **GPU**: 8x H100 SXM (80GB) - 脚本默认配置
- **内存**: 2TB+ 系统内存
- **存储**: 100GB+ 快速存储 (数据 + 日志)

### 支持的硬件
- **H100**: 989 TFLOPS (bfloat16) - 推荐
- **A100**: 312 TFLOPS (bfloat16) - 支持
- **B200**: 2500 TFLOPS (估计) - 自动检测

## 验证

### MuP 正确性验证

使用坐标检查功能验证 MuP 实现的正确性。正确的 MuP 实现应在不同模型宽度间展现稳定的激活幅度。

```bash
# 运行宽度扫描验证
bash scripts/run_mup_width_group_scaling.sh 256 "512 768 1024" 1000

# 检查生成的坐标检查图
# 正确实现: 激活统计在不同宽度间保持稳定
# 错误实现: 激活幅度随宽度显著变化
```

## 使用建议

### 开发和测试
```bash
# 使用较小配置快速测试
python train.py --width 256 --iterations 100 --use-wandb false
```

### 研究实验
```bash
# 启用完整监控和坐标检查
python train.py --mup --width 1024 --base-width 768 \
  --coord-check --use-wandb true --wandb-project "research"
```

### 生产训练
```bash
# 8xH100 分布式训练
bash scripts/run_mup_width_group_scaling.sh 768 "1024 1536 2048" 1750
```

## 故障排除

### 常见问题

1. **数据未找到**: 确保运行了数据下载脚本
2. **CUDA 内存不足**: 减少批大小或序列长度
3. **坐标检查失败**: 检查 MuP 配置和基准宽度设置
4. **性能缓慢**: 禁用昂贵的监控指标

### 调试模式

```bash
# 启用详细日志和频繁监控
python train.py --width 512 --iterations 10 \
  --log-every 1 --coord-check-every 5 --use-wandb true
```

## 参考资料

- [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)
- [Modded-NanoGPT 仓库](https://github.com/KellerJordan/modded-nanogpt)
- [MuP 仓库](https://github.com/microsoft/mup)
- [μP 理论与实践 (kexue.fm)](https://kexue.fm/archives/10795)

## 性能分析文档

详细的监控系统性能影响分析请参阅：[METRICS_ANALYSIS.md](METRICS_ANALYSIS.md)

---

**注意**: 这是研究用途的实现，专注于 MuP 理论验证和缩放实验。生产环境使用请充分测试。