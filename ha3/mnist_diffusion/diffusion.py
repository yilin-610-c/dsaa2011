from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


def extract(values: torch.Tensor, timesteps: torch.Tensor, x_shape) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule: str, training_t: int, device: torch.device | str = "cpu"
) -> torch.Tensor:
    if schedule in {"linear", "scaled_linear"}:
        beta_start = 1e-4 * 1000.0 / training_t
        beta_end = 0.02 * 1000.0 / training_t
        return torch.linspace(beta_start, beta_end, training_t, device=device).clamp(
            max=0.999
        )
    if schedule == "cosine":
        s = 0.008
        steps = torch.arange(training_t + 1, device=device, dtype=torch.float32)
        x = steps / training_t
        alpha_bar = torch.cos(((x + s) / (1.0 + s)) * torch.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        return betas.clamp(min=1e-8, max=0.999)
    raise ValueError(f"Unknown schedule: {schedule}")


@dataclass
class DiffusionSchedule:
    schedule: str
    training_t: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor
    alpha_bars_prev: torch.Tensor
    sqrt_alpha_bars: torch.Tensor
    sqrt_one_minus_alpha_bars: torch.Tensor
    sqrt_recip_alpha_bars: torch.Tensor
    sqrt_recipm1_alpha_bars: torch.Tensor
    posterior_variance: torch.Tensor
    posterior_mean_coef1: torch.Tensor
    posterior_mean_coef2: torch.Tensor

    @property
    def alpha_bar_T(self) -> float:
        return float(self.alpha_bars[-1].detach().cpu())

    def to(self, device: torch.device | str):
        return build_schedule(self.schedule, self.training_t, device=device)


def build_schedule(
    schedule: str, training_t: int, device: torch.device | str = "cpu"
) -> DiffusionSchedule:
    betas = make_beta_schedule(schedule, training_t, device=device).float()
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_variance = posterior_variance.clamp(min=1e-20)
    return DiffusionSchedule(
        schedule=schedule,
        training_t=training_t,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        alpha_bars_prev=alpha_bars_prev,
        sqrt_alpha_bars=torch.sqrt(alpha_bars),
        sqrt_one_minus_alpha_bars=torch.sqrt(1.0 - alpha_bars),
        sqrt_recip_alpha_bars=torch.sqrt(1.0 / alpha_bars),
        sqrt_recipm1_alpha_bars=torch.sqrt(1.0 / alpha_bars - 1.0),
        posterior_variance=posterior_variance,
        posterior_mean_coef1=betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars),
        posterior_mean_coef2=(1.0 - alpha_bars_prev)
        * torch.sqrt(alphas)
        / (1.0 - alpha_bars),
    )


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    return extract(schedule.sqrt_alpha_bars, timesteps, x0.shape) * x0 + extract(
        schedule.sqrt_one_minus_alpha_bars, timesteps, x0.shape
    ) * noise


def predict_x0_from_eps(
    xt: torch.Tensor,
    timesteps: torch.Tensor,
    eps: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    return extract(schedule.sqrt_recip_alpha_bars, timesteps, xt.shape) * xt - extract(
        schedule.sqrt_recipm1_alpha_bars, timesteps, xt.shape
    ) * eps


@torch.no_grad()
def ddpm_sample(
    model,
    schedule: DiffusionSchedule,
    shape,
    device,
    initial_noise: Optional[torch.Tensor] = None,
    clip_x0: bool = True,
) -> torch.Tensor:
    model.eval()
    x = (
        torch.randn(shape, device=device)
        if initial_noise is None
        else initial_noise.to(device).clone()
    )
    for step in reversed(range(schedule.training_t)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        eps = model(x, t)
        x0 = predict_x0_from_eps(x, t, eps, schedule)
        if clip_x0:
            x0 = x0.clamp(-1.0, 1.0)
        mean = extract(schedule.posterior_mean_coef1, t, x.shape) * x0 + extract(
            schedule.posterior_mean_coef2, t, x.shape
        ) * x
        if step > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(extract(schedule.posterior_variance, t, x.shape)) * noise
        else:
            x = mean
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def ddim_sample(
    model,
    schedule: DiffusionSchedule,
    shape,
    device,
    sampling_steps: int = 50,
    initial_noise: Optional[torch.Tensor] = None,
    eta: float = 0.0,
    clip_x0: bool = True,
) -> torch.Tensor:
    model.eval()
    x = (
        torch.randn(shape, device=device)
        if initial_noise is None
        else initial_noise.to(device).clone()
    )
    timesteps = torch.linspace(
        schedule.training_t - 1, 0, sampling_steps, device=device
    ).long()
    timesteps = torch.unique_consecutive(timesteps)
    for i, step_tensor in enumerate(timesteps):
        step = int(step_tensor.item())
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        eps = model(x, t)
        x0 = predict_x0_from_eps(x, t, eps, schedule)
        if clip_x0:
            x0 = x0.clamp(-1.0, 1.0)

        prev_step = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
        alpha_t = schedule.alpha_bars[step]
        alpha_prev = (
            schedule.alpha_bars[prev_step]
            if prev_step >= 0
            else torch.tensor(1.0, device=device)
        )
        if eta > 0.0 and prev_step >= 0:
            sigma = (
                eta
                * torch.sqrt((1 - alpha_prev) / (1 - alpha_t))
                * torch.sqrt(1 - alpha_t / alpha_prev)
            )
            noise = torch.randn_like(x)
        else:
            sigma = torch.tensor(0.0, device=device)
            noise = 0.0
        direction = torch.sqrt((1.0 - alpha_prev - sigma**2).clamp(min=0.0)) * eps
        x = torch.sqrt(alpha_prev) * x0 + direction + sigma * noise
    return x.clamp(-1.0, 1.0)

