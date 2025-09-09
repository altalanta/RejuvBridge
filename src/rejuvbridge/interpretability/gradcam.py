from typing import Tuple
import torch
import torch.nn as nn


def grad_cam(model: nn.Module, x: torch.Tensor, target_layer: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Minimal Grad-CAM stub returning heatmap same HxW as input."""
    model.eval()
    feats = None

    def hook_fn(_, __, output):
        nonlocal feats
        feats = output

    h = target_layer.register_forward_hook(hook_fn)
    x = x.requires_grad_(True)
    out = model(x)
    cls = out.argmax(dim=1).sum()
    cls.backward()
    grads = x.grad
    h.remove()
    heat = grads.abs().mean(dim=1, keepdim=True)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    return out.detach(), heat.detach()

