好主意，但先把“物理事实”说清楚：**五阶 Newton–Schulz（NS5）本身只是在做极分解的“正交因子”（≈ $UV^\top$）**，它不会直接给你奇异值（也就不是直接给谱范数）。要拿到谱范数 $\sigma_{\max}(W)$，常规且高效的是**对 $W^\top W$** 做幂迭代（PyTorch 的 `spectral_norm` 也是这么干的）([PyTorch Docs][1], [GitHub][2])。
不过既然你希望“沿用 Muon 的 NS5 流程、且能在 bf16 上稳定跑”，我们可以这样做：

1. 用 NS5 得到 **极分解的正交因子** $Q \approx \text{polar}(W)$（完全 GPU matmul 友好，bf16 可行，这是 Muon 的核心做法）([kellerjordan.github.io][3])。
2. 由极分解 $W = QH$ 可知 $H = Q^\top W \approx (W^\top W)^{1/2}$，其**特征值就是 $W$** 的奇异值，于是对 $H$ 做**幂迭代**即可估计 $\sigma_{\max}(W)$ —— 实现上我们不显式构造 $H$，而是用线性算子 $v \mapsto Q^\top (Wv)$ 来做幂迭代（两次 matmul）([Nick Higham][4], [arXiv][5])。

> 注：Muon 为了速度，NS5 的系数特意调成“**不严格收敛到 1** 但**斜率大**”的多项式（$a,b,c=(3.4445,-4.7750,2.0315)$），这样 $UV^\top$ 会被近似成 $US'V^\top$ 且 $S'\in[0.68,1.13]$ 一带；对**优化**影响很小，但对**高精度谱范数估计**会带来几十个点的偏差。如果你更在意精度，可切到“收敛型”NS3（经典 $X\_{k+1}=\tfrac12 X_k(3I-X_k^\top X_k)$），代价是多几步迭代。下面代码两种都支持（`mode="ns5-fast"` 或 `mode="ns3-accurate"`）([GitHub][6])。

---

### 直接可用的实现（bf16 友好，不干扰训练精度）

```python
from typing import Dict, Optional, Iterable
import contextlib
import torch
from torch import nn, Tensor

# ---------- Newton–Schulz building blocks (bf16-friendly, GPU matmul only) ----------

@torch.no_grad()
def _ns5_zeroth_power(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Quintic Newton–Schulz used in Muon to approx the orthogonal polar factor.
    Uses the (a,b,c) tuned for large slope at 0 (non-convergent to 1 but fast).
    Ref: Keller Jordan's Muon writeup & modded-nanogpt README.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)  # Muon coefficients
    # Normalize to keep singular values in [0,1]-ish
    X = G.to(dtype=torch.bfloat16)
    # Use Fro norm; for tall/flat handling we follow Muon trick (transpose to tall)
    if X.size(0) > X.size(1):
        X = X.mT
        transposed = True
    else:
        transposed = False
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X  # keep bf16; caller may cast

@torch.no_grad()
def _ns3_polar(G: Tensor, steps: int = 8, eps: float = 1e-7) -> Tensor:
    """
    Classic convergent Newton–Schulz for polar factor:
      X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k)
    After pre-scaling. Slower but better accuracy for polar(Q).
    """
    assert G.ndim == 2
    X = G.to(dtype=torch.bfloat16)
    if X.size(0) > X.size(1):
        X = X.mT
        transposed = True
    else:
        transposed = False
    X = X / (X.norm() + eps)
    I = torch.eye(X.size(-1), device=X.device, dtype=X.dtype)
    for _ in range(steps):
        XtX = X.mT @ X
        X = 0.5 * (X @ (3*I - XtX))
    if transposed:
        X = X.mT
    return X

@torch.no_grad()
def _polar_factor(G: Tensor, *, mode: str = "ns5-fast") -> Tensor:
    if mode == "ns5-fast":
        return _ns5_zeroth_power(G, steps=5)
    elif mode == "ns3-accurate":
        return _ns3_polar(G, steps=8)
    else:
        raise ValueError(f"unknown mode={mode}")

# ---------- Spectral norm via polar + power iteration on H ≈ (W^T W)^{1/2} ----------

@torch.no_grad()
def _spectral_norm_via_polar_power(W: Tensor, *, mode: str, power_iters: int = 7) -> float:
    """
    Estimate sigma_max(W) using:
      1) Q ≈ polar(W) by Newton–Schulz (bf16-friendly)
      2) H = Q^T W ≈ (W^T W)^{1/2}; run power-iteration on linear op v -> Q^T (W v)
    Memory-lean: never explicitly forms H.
    """
    device = W.device
    # Flatten conv/ND to (out_features, in_features)
    if W.ndim > 2:
        W2 = W.reshape(W.size(0), -1)
    else:
        W2 = W

    # Build Q in bf16 (GPU-friendly). Keep W matmuls in bf16 as well.
    Q = _polar_factor(W2, mode=mode)  # bf16
    n = W2.size(1)
    # Power iteration on symmetric PSD linear operator H(v) = Q^T (W v)
    # Work vectors in bf16 for matmul, but norms/inner-products in fp32
    v = torch.randn(n, device=device, dtype=torch.bfloat16)
    v = v / (v.float().norm() + 1e-12)

    for _ in range(max(1, power_iters)):
        y = W2.to(torch.bfloat16) @ v              # shape (m,)
        z = Q.mT @ y                               # shape (n,) == H @ v
        v = z / (z.float().norm() + 1e-12)

    # Rayleigh quotient on the last (v, Hv)
    y = W2.to(torch.bfloat16) @ v
    Hv = Q.mT @ y
    num = (v.float() * Hv.float()).sum()
    den = (v.float() * v.float()).sum()
    lam = (num / (den + 1e-20)).item()            # eigenvalue of H
    # H ≈ (W^T W)^{1/2} => eigenvalues of H are singular values of W
    sigma = max(lam, 0.0)
    return float(sigma)

# ---------- Public API ----------

@torch.no_grad()
def compute_weight_spectral_norms(
    model: nn.Module,
    names_filter: Optional[Iterable[str]] = None,
    mode: str = "ns5-fast",          # or "ns3-accurate"
    power_iters: int = 7
) -> Dict[str, float]:
    """
    Compute per-weight spectral norms during bf16/fp16 mixed-precision training
    without changing training dtype. Heavy ops done in bf16; reductions in fp32.

    mode:
      - "ns5-fast": Muon-style quintic Newton–Schulz (fast, small bias possible)
      - "ns3-accurate": classic convergent NS3 (slower, more accurate polar factor)

    Returns:
      {parameter_name: approx_sigma_max}
    """
    specs: Dict[str, float] = {}

    # ensure we don't get autocast surprises from the training region
    maybe_no_autocast = (
        torch.amp.autocast(device_type='cuda', enabled=False)
        if torch.cuda.is_available() else contextlib.nullcontext()
    )
    with maybe_no_autocast:
        for name, p in model.named_parameters():
            if (names_filter is not None) and (name not in names_filter):
                continue
            if p.ndim < 2 or 'weight' not in name:
                continue
            W = p.detach()  # never touches autograd graph
            try:
                sigma = _spectral_norm_via_polar_power(W, mode=mode, power_iters=power_iters)
            except Exception:
                # Robust fallback: plain power iteration on W^T W (still bf16 matmuls)
                # This matches PyTorch spectral_norm's math, but stays off matrix_norm().
                if W.ndim > 2:
                    W2 = W.reshape(W.size(0), -1)
                else:
                    W2 = W
                m, n = W2.shape
                v = torch.randn(n, device=W2.device, dtype=torch.bfloat16)
                v = v / (v.float().norm() + 1e-12)
                for _ in range(max(3, power_iters)):
                    z = W2.mT @ (W2.to(torch.bfloat16) @ v)  # (n,)
                    v = z / (z.float().norm() + 1e-12)
                # Rayleigh: v^T (W^T W) v
                z = W2.mT @ (W2.to(torch.bfloat16) @ v)
                num = (v.float() * z.float()).sum()
                den = (v.float() * v.float()).sum()
                sigma = float(torch.sqrt((num / (den + 1e-20)).clamp_min(0)).item())
            specs[name] = sigma
    return specs
```

**怎么用**

```python
# 仅观测注意力和MLP主干层，每隔 N 步打一遍
names = {
    "transformer.blocks.0.attn.out_proj.weight",
    "transformer.blocks.0.mlp.down_proj.weight",
    # ... 你关心的更多层
}
specs = compute_weight_spectral_norms(model, names_filter=names,
                                      mode="ns5-fast", power_iters=7)
for k, v in specs.items():
    print(f"[spec] {k}: {v:.4f}")
```

---

### 为什么这套做法靠谱（而且 bf16 友好）

* **Muon 的 NS5**：用只含 matmul 的多项式迭代来近似极分解正交因子，官方写法本就直接在 **bf16** 上跑（文中与仓库都这么实现），且为性能选择了“高斜率的五阶多项式”与“必要时转置成高的矩阵再做迭代”的 trick。我们照搬这些做法来保证在你现有 Muon 训练里**无缝观测**。([kellerjordan.github.io][3], [GitHub][6])
* **极分解与谱范数**：$W=QH$，其中 $H=(W^\top W)^{1/2}$ 的特征值就是奇异值，所以**对 $H$** 做幂迭代就能取到 $\sigma_{\max}(W)$。我们通过线性算子 $v\mapsto Q^\top(Wv)$ 避免显式构造 $H$（内存友好）。([Nick Higham][4], [arXiv][5])
* **备用方案**：若 NS5 因任意数值问题失败，备份到“**直接对 $W^\top W$** 的幂迭代”，这与 PyTorch `spectral_norm` 的思想一致，但不依赖 `matrix_norm`，因此不会踩 bf16 不支持的问题。([PyTorch Docs][1])

---

### 小结与取舍

* 想**速度优先**、和 Muon 路线一致：`mode="ns5-fast"`（可能有轻微偏差，但对监控趋势、报警阈值足够）。
* 想**精度更高**：`mode="ns3-accurate"`（多几步迭代，更接近严格的 $UV^\top$）。
* 极限稳妥：直接用**幂迭代 $W^\top W$**（已在 fallback 里提供）。

如果你愿意，我还能把它打包成 **TensorBoard/CSV 回调**（含分布式只在 rank0 记录、分频抽样、层名通配）——顺手加一个“**谱范数/层宽**”与“**谱范数随训练步曲线**”的小面板，方便你对 Muon 的谱条件做在线体检 🩺。

---

**参考**

* Muon 设计、NS5 代码与 bf16 实践（含“非收敛但斜率大”的系数）：Keller Jordan 博客与 `modded-nanogpt` 说明。([kellerjordan.github.io][3], [GitHub][6])
* PyTorch `spectral_norm` 原理：幂迭代近似最大奇异值。([PyTorch Docs][1], [GitHub][2])
* 极分解与 NS 在深度学习/极分解中的背景与 GPU 友好算法（Polar Express；加速 NS 的近期论文；Higham/Chen 等经典资料）。([arXiv][7], [Nick Higham][4], [jiechenjiechen.github.io][8])

[1]: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.spectral_norm.html?utm_source=chatgpt.com "torch.nn.utils.spectral_norm"
[2]: https://github.com/pytorch/pytorch/blob/main/torch/nn/utils/spectral_norm.py?utm_source=chatgpt.com "pytorch/torch/nn/utils/spectral_norm.py at main"
[3]: https://kellerjordan.github.io/posts/muon/ "Muon: An optimizer for hidden layers in neural networks | Keller Jordan blog"
[4]: https://nhigham.com/2020/07/28/what-is-the-polar-decomposition/comment-page-1/?utm_source=chatgpt.com "What is the Polar Decomposition? - Nick Higham"
[5]: https://arxiv.org/pdf/2506.10935?utm_source=chatgpt.com "Accelerating Newton-Schulz Iteration for Orthogonalization ..."
[6]: https://github.com/KellerJordan/modded-nanogpt "GitHub - KellerJordan/modded-nanogpt: NanoGPT (124M) in 3 minutes"
[7]: https://arxiv.org/html/2505.16932v2?utm_source=chatgpt.com "The Polar Express: Optimal Matrix Sign Methods and Their ..."
[8]: https://jiechenjiechen.github.io/pub/sign.pdf?utm_source=chatgpt.com "A Stable Scaling of Newton-Schulz for Improving the Sign ..."
