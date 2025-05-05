import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer


class AMS(Optimizer):
    """
    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.999),
        scaling: float = 1.0,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta_1 parameter: {betas[0]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta_2 parameter: {betas[1]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(
                f"Invalid beta_3 parameter: {betas[2]} - should be in [0.0, 1.0)"
            )
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {
            "lr": lr,
            "betas": betas,
            "scaling": scaling,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)
        self.init_lr = lr

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                beta1, beta2, beta3 = group["betas"]
                scaling = group["scaling"]

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    if beta3 > 0:
                        # Exponential moving average of sign of gradient values
                        state["exp_sign"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if beta3 > 0 and beta3 != beta1 and scaling != 1:
                    exp_sign = state["exp_sign"]

                state["step"] += 1

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                if (
                    beta3 > 0 and beta3 != beta1 and scaling != 1
                ):  # beta3 == beta1 is equivalent to Adam
                    exp_sign.mul_(beta3).add_(grad, alpha=(1.0 - beta3))
                elif beta3 == 0:
                    exp_sign = grad

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                if beta3 == beta1 or scaling == 1:
                    # Adam
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                elif beta3 == 0 and scaling == -1:
                    # Grams
                    grad.sign_().mul_(exp_avg.abs())
                    p.addcdiv_(grad, denom, value=-step_size)
                elif beta3 == 0 and scaling != -1:
                    # Gradient AMS (GAMS)
                    phi = (exp_avg.sign() * grad.sign() > 0).to(grad.dtype)
                    psi = ((1 - phi) * scaling).to(grad.dtype)
                    total_mask = (phi + psi) * (
                        phi.numel() / (phi.sum() + psi.sum().abs() + 1)
                    )
                    p.addcdiv_(exp_avg * total_mask, denom, value=-step_size)
                else:
                    # AMS
                    phi = (exp_avg.sign() * exp_sign.sign() > 0).to(grad.dtype)
                    psi = ((1 - phi) * scaling).to(grad.dtype)
                    total_mask = (phi + psi) * (
                        phi.numel() / (phi.sum() + psi.sum().abs() + 1)
                    )
                    p.addcdiv_(exp_avg * total_mask, denom, value=-step_size)

        return loss
