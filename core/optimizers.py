"""
Advanced optimizers from modded-nanogpt with MuP support.

Includes:
- Muon optimizer with Newton-Schulz orthogonalization
- DistAdam optimizer with distributed training support
- Both optimizers support MuP scaling when used with proper base shapes
"""

import torch
import torch.distributed as dist
from torch import Tensor
from typing import List


# -----------------------------------------------------------------------------
# Muon optimizer (from modded-nanogpt)

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Warning: This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        # Efficient systems-wise implementation of step developed by @YouJiacheng,
        # @KonstantinWilleke, @alexrgilbert, @adricarda, @tuttyfrutyee, @vdlad,
        # @ryanyang0, and @vagrawal.
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue
                
            grad = torch.empty_like(params[-1])
            grad_pad = [param.grad for param in params] + [torch.zeros_like(params[-1])] * world_size
            for base_i in range(0, len(params), world_size):
                if base_i + rank < len(params):
                    grad = params[base_i + rank].grad
                # This gives strange dynamo warnings
                if dist.is_initialized():
                    reduce_scatter_futures.append(dist.reduce_scatter(grad, grad_pad[base_i:base_i + world_size], op=dist.ReduceOp.AVG, async_op=True).get_future())

        idx = 0
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue
                
            params_pad = params + [torch.empty_like(params[-1])] * world_size
            momentum = group["momentum"]
            for base_i in range(0, len(params), world_size):
                if dist.is_initialized():
                    reduce_scatter_futures[idx].wait()
                if base_i + rank < len(params):
                    p = params[base_i + rank]
                    grad = p.grad
                    eff_lr = group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5 * getattr(p, "lr_mul", 1.0)
                    eff_weight_decay = group["lr"] * group["weight_decay"] * getattr(p, "wd_mul", 1.0)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                    p.mul_(1 - eff_weight_decay)
                    momentum_buffer.lerp_(grad, 1 - momentum)
                    grad = grad.lerp_(momentum_buffer, momentum)
                    v = zeropower_via_newtonschulz5(grad.bfloat16(), 5)
                    p.add_(other=v, alpha=-eff_lr)
                idx += 1
                if dist.is_initialized():
                    all_reduce_futures.append(dist.all_gather(params_pad[base_i:base_i + world_size], params_pad[base_i + rank], async_op=True).get_future())
        
        if dist.is_initialized():
            torch.futures.collect_all(all_reduce_futures).wait()


class DistAdam(torch.optim.Optimizer):
    """Distributed Adam optimizer from modded-nanogpt."""
    
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        params = list(params)
        sizes = {p.shape for p in params}
        # create one buffer per unique parameter-size
        param_groups = []
        for size in sizes:
            group_params = [p for p in params if p.shape == size]
            param_groups.append(dict(params=group_params))
        super().__init__(param_groups, defaults)
        # DistributedAdam implementation by @vagrawal

    @torch.compile
    @torch.no_grad()
    def step(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        reduce_scatter_futures: list[torch.Future] = []
        all_reduce_futures: list[torch.Future] = []
        grad_slices = []
        
        for group in self.param_groups:
            params: list[Tensor] = group["params"]
            if not params:
                continue
                
            grad = torch.empty_like(params[-1])
            for base_i in range(len(params)):
                grad = params[base_i].grad
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                if dist.is_initialized():
                    reduce_scatter_futures.append(dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future())
                else:
                    grad_slice.copy_(grad[:rank_size])
                grad_slices.append(grad_slice)

        idx = 0
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            params = group['params']
            for base in range(len(params)):
                if dist.is_initialized():
                    reduce_scatter_futures[idx].wait()
                p = params[base]
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
                lr = group['lr'] * getattr(p, "lr_mul", 1.0)
                state = self.state[p]
                g_slice = grad_slices[idx]
                # State init
                if not state:
                    state['step'] = torch.tensor(0, dtype=torch.int64, device=p.device)
                    state['exp_avg'] = torch.zeros_like(p_slice)
                    state['exp_avg_sq'] = torch.zeros_like(p_slice)
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                # weight decay
                if wd != 0:
                    eff_weight_decay = lr * wd * getattr(p, "wd_mul", 1.0)
                    p_slice.mul_(1 - eff_weight_decay)
                # update running averages
                exp_avg.mul_(beta1).add_(g_slice, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_slice, g_slice, value=1 - beta2)
                # bias corrections
                bias1 = 1 - beta1 ** t
                bias2 = 1 - beta2 ** t
                # compute step
                denom = exp_avg_sq.sqrt().add_(eps)
                step_size = lr * (torch.sqrt(bias2) / bias1)
                update = exp_avg.div(denom).mul_(step_size)
                p_slice.add_(other=update, alpha=-1.0)
                idx += 1
                if dist.is_initialized():
                    all_reduce_futures.append(dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future())
        
        if dist.is_initialized():
            torch.futures.collect_all(all_reduce_futures).wait()


# -----------------------------------------------------------------------------
# MuP-aware optimizer factories

def create_muon_optimizer(model_parameters, lr: float = 0.05, weight_decay: float = 0.0, momentum: float = 0.95):
    """Create Muon optimizer with proper parameter filtering."""
    # Separate matrix-like parameters (2D or higher) from others
    matrix_params = []
    other_params = []
    
    for param in model_parameters:
        if param.dim() >= 2:
            # Check if this should use Muon based on size and type
            if param.numel() >= 128:  # Only large matrices benefit from Muon
                matrix_params.append(param)
            else:
                other_params.append(param)
        else:
            other_params.append(param)
    
    # Create optimizers
    optimizers = []
    if matrix_params:
        muon_opt = Muon(matrix_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        optimizers.append(muon_opt)
    
    if other_params:
        # Use Adam for embeddings, scalars, small matrices, etc.
        adam_opt = torch.optim.AdamW(other_params, lr=lr*0.16, weight_decay=0.0, betas=(0.8, 0.95), eps=1e-10)
        optimizers.append(adam_opt)
    
    return optimizers



def step_optimizers(optimizers: List[torch.optim.Optimizer]):
    """Step all optimizers in the list."""
    for opt in optimizers:
        opt.step()


def zero_grad_optimizers(optimizers: List[torch.optim.Optimizer]):
    """Zero gradients for all optimizers."""
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)


def get_lr_schedulers(optimizers: List[torch.optim.Optimizer], num_iterations: int, cooldown_frac: float = 0.45):
    """Create learning rate schedulers for optimizers."""
    def get_lr(step: int):
        x = step / num_iterations  # progress in training
        assert 0 <= x < 1
        if x < 1 - cooldown_frac:
            return 1.0
        else:
            w = (1 - x) / cooldown_frac
            return w * 1.0 + (1 - w) * 0.1
    
    schedulers = []
    for opt in optimizers:
        # Store initial learning rates
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, get_lr)
        schedulers.append(scheduler)
    
    return schedulers


def apply_lr_schedule(optimizers: List[torch.optim.Optimizer], step: int, num_iterations: int, cooldown_frac: float = 0.45):
    """Apply learning rate schedule to optimizers."""
    x = step / num_iterations
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        lr_mult = 1.0
    else:
        w = (1 - x) / cooldown_frac
        lr_mult = w * 1.0 + (1 - w) * 0.1
    
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lr_mult


def apply_momentum_warmup(optimizers: List[torch.optim.Optimizer], step: int, warmup_steps: int = 300):
    """Apply momentum warmup to Muon optimizers."""
    frac = min(step / warmup_steps, 1)
    target_momentum = (1 - frac) * 0.85 + frac * 0.95
    
    for opt in optimizers:
        if isinstance(opt, Muon):
            for group in opt.param_groups:
                group["momentum"] = target_momentum