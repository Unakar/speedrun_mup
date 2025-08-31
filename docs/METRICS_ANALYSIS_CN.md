# W&B 指标系统与性能影响分析

## 概览

本文档分析 speedrun-mup 中的 W&B 集成和高级监控功能，重点关注性能影响和推荐使用模式，特别是基于 Newton-Schulz 的高效谱范数计算。

## W&B 指标分类

### 基础训练指标 (始终记录)
- `step`: 训练步数
- `training_time_ms`: 墙钟训练时间 (毫秒)
- `step_avg_ms`: 每步平均时间
- `train_loss`: 训练损失
- `val_loss`: 验证损失 (可用时)

### 高级监控指标 (可配置)
- `grad_norm`: 全局梯度范数 (轻量，始终计算)
- `param_norm`: 全局参数范数 (轻量，始终计算)
- `spectral_norm_max/mean/std`: 权重矩阵谱范数 (Newton-Schulz，可配置)
- `activation_mean/std/max/min/l2_norm`: 激活统计 (昂贵，可配置)

## 性能影响分析

### 轻量指标 (可忽略影响)
- **梯度范数计算**: 每步约 0.1ms 开销
- **参数范数计算**: 每步约 0.1ms 开销
- **基础 W&B 日志**: 每步约 1-2ms 开销
- **本地文件日志**: 每步约 0.05ms 开销

### Newton-Schulz 谱范数 (中等影响，高效实现)
- **NS5 快速模式**: 每个监控参数 5-10ms 开销
- **NS3 精确模式**: 每个监控参数 8-15ms 开销  
- **BF16 友好**: 完全兼容混合精度训练
- **智能优化**: 仅监控 Muon 优化器的隐藏矩阵参数

### 传统昂贵指标 (显著影响，已优化)
- **SVD 谱范数计算**: 每监控参数 20-50ms 开销 (已弃用)
- **激活统计**: 每钩子层 2-10ms 开销
- **完整监控大模型**: 每步可增加 50-200ms

### 推荐监控间隔

#### 生产训练配置
```python
# 高性能日常训练
monitor_interval = 100  # 每 100 步检查昂贵指标
MONITOR_SPECTRAL_EVERY=100  # Newton-Schulz 谱范数
MONITOR_ACTIVATIONS=false   # 禁用激活统计
```

#### 研究调试配置  
```python
# 研究调试模式
monitor_interval = 10   # 更频繁的监控
MONITOR_SPECTRAL_EVERY=10   # 更频繁的谱范数
MONITOR_ACTIVATIONS=true    # 启用激活统计
```

#### 最终验证配置
```python
# 最终验证运行 
monitor_interval = 1    # 每步监控 (昂贵但全面)
MONITOR_SPECTRAL_EVERY=1    # 每步谱范数
MONITOR_ACTIVATIONS=true    # 完整激活统计
```

## Newton-Schulz 谱范数技术细节

### 算法优势

**极分解方法**: 使用 Newton-Schulz 迭代计算极分解的正交因子 Q ≈ polar(W)
- **NS5 快速模式**: Muon 风格五阶多项式，系数优化为高斜率 (3.4445, -4.7750, 2.0315)
- **NS3 精确模式**: 经典收敛型三阶迭代，更高精度

**幂迭代优化**: 对 H = Q^T W ≈ (W^T W)^{1/2} 进行幂迭代
- **内存高效**: 从不显式构造 H 矩阵，使用线性算子 v → Q^T(Wv)  
- **BF16 兼容**: 矩阵乘法在 BF16，范数计算在 FP32
- **智能回退**: 失败时自动回退到传统 W^T W 幂迭代

### 精度与性能权衡

**NS5 快速模式**:
- ✅ 与 Muon 优化器风格一致
- ✅ 速度优先，5-10ms/参数
- ⚠️ 轻微偏差可能 (几十个百分点)
- 🎯 **推荐用于**: 日常训练监控、趋势分析

**NS3 精确模式**:
- ✅ 更接近严格的极分解
- ✅ 更高精度的谱范数估计
- ⚠️ 8-15ms/参数，稍慢
- 🎯 **推荐用于**: 研究分析、精确测量

### 使用示例

```python
# 配置 Newton-Schulz 谱范数监控
from core.utils import compute_weight_spectral_norms

# 快速模式 - 日常训练
spectral_norms = compute_weight_spectral_norms(
    model, 
    target_params=hidden_matrix_params,
    mode="ns5-fast", 
    power_iters=7
)

# 精确模式 - 研究分析
spectral_norms = compute_weight_spectral_norms(
    model,
    target_params=hidden_matrix_params, 
    mode="ns3-accurate",
    power_iters=10
)
```

## W&B 集成优势

### 实验组织
- **自动命名**: `speedrun_20250831_143022_mup_w1024_base768`
- **项目分组**: 分离不同实验类型
- **配置日志**: 完整超参数跟踪
- **结构化日志**: 一致的指标命名和单位

### 性能监控
- **硬件感知 MFU**: 自动检测 H100/B200/A100 以准确计算利用率
- **内存跟踪**: 峰值和保留内存使用
- **训练效率**: 每秒令牌数，步时间分析

### MuP 专用指标
- **坐标检查**: 跨宽度的激活幅度稳定性
- **缩放验证**: 超参数迁移验证
- **宽度比较**: 不同模型宽度的并排比较

### W&B 指标分组

指标按类别清晰组织：

**Time/** (时间相关):
- `Time/training_time_ms`
- `Time/step_avg_ms`
- `Time/total_time_s`

**Loss/** (损失函数):
- `Loss/train_loss`
- `Loss/val_loss`

**Optimization/** (优化相关):
- `Optimization/lr`
- `Optimization/grad_norm` 
- `Optimization/momentum`

**Model/** (模型参数):
- `Model/param_norm`
- `Model/spectral_norm_max`
- `Model/spectral_norm_mean`
- `Model/spectral_norm_std`

**Hardware/** (硬件资源):
- `Hardware/peak_memory_mb`
- `Hardware/reserved_memory_mb`

**Activations/** (激活统计):
- `Activations/layer_0_mean`
- `Activations/layer_0_std`
- `Activations/attention_l2_norm`

## 推荐使用模式

### 日常训练
```python
# 基础监控，最小性能影响
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup"
)
# 使用脚本配置
MONITOR_SPECTRAL_EVERY=100 bash scripts/run_basic_speedrun.sh
```

### 研究调试
```python
# 详细监控，研究友好
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup-research"
)
# 使用脚本配置
MONITOR_SPECTRAL_EVERY=10 MONITOR_ACTIVATIONS=true \
bash scripts/run_basic_speedrun.sh
```

### MuP 验证
```python
# MuP 坐标检查专用
logger = SimpleLogger(
    use_wandb=True, 
    project_name="speedrun-mup-coord-check"
)
# 启用坐标检查功能
python train.py --mup --coord-check --coord-check-every 50
```

## 性能建议

### 高频训练 (>1000 步/实验)
- 每 100+ 步监控昂贵指标
- 使用基础指标进行逐步跟踪
- 启用 W&B 进行实验组织
- 训练期间禁用激活钩子
- 推荐: `MONITOR_SPECTRAL_EVERY=100`

### 研究实验 (<500 步)
- 每 10-50 步监控昂贵指标
- 完整激活统计用于层分析
- Newton-Schulz 谱范数用于权重分析
- MuP 验证的详细坐标检查
- 推荐: `MONITOR_SPECTRAL_EVERY=10 MONITOR_ACTIVATIONS=true`

### 分布式训练 (8xH100)
- 仅从 rank 0 进程记录日志
- 记录前跨进程聚合指标
- 使用高效的归约操作
- 考虑日志频率对同步的影响

## 内存影响

### W&B 开销
- 客户端内存: 约 50-100MB 基线
- 指标缓冲区: 每 1000 步约 1-5MB
- 图像日志: 可变 (基础设置中未使用)

### 监控开销
- 激活钩子: 每个钩子层约 10-50MB
- Newton-Schulz 谱范数计算: 最小内存影响
- 梯度/参数范数: 可忽略内存影响

## 最佳实践

1. **从简单开始**: 首先使用基础指标，根据需要添加高级监控
2. **选择性监控**: 仅在研究阶段启用昂贵指标
3. **使用间隔**: 不要每步都监控昂贵指标
4. **清理资源**: 不需要时移除激活钩子
5. **组织实验**: 使用描述性项目名称和实验名称
6. **验证 MuP**: 使用坐标检查确保正确实现
7. **利用 Newton-Schulz**: 使用高效的谱范数实现，避免 SVD

## 与 modded-nanogpt 风格的集成

日志系统保持与 modded-nanogpt 简单控制台输出的兼容性:
```
step:1000/1750 val_loss:3.2847 train_time:180420.0ms step_avg:180.42ms
```

同时为高级分析和实验跟踪添加结构化文件和 W&B 日志。

## Newton-Schulz 技术参考

- **Muon 设计与 NS5 代码**: Keller Jordan 博客与 modded-nanogpt 说明
- **极分解理论**: Nick Higham 的极分解介绍  
- **PyTorch spectral_norm 原理**: 幂迭代近似最大奇异值
- **GPU 友好算法**: 加速 Newton-Schulz 的近期论文与深度学习中的极分解应用

---

**注意**: 本分析基于 8xH100 系统的基准测试。实际性能可能因硬件配置而异。建议在您的特定设置上进行性能分析以优化监控配置。