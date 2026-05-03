import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half, device=timesteps.device, dtype=torch.float32
    )
    exponent = exponent / max(half - 1, 1)
    freqs = torch.exp(exponent)
    args = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class MNISTUNet(nn.Module):
    def __init__(self, base_channels: int = 64, time_dim: int = 256):
        super().__init__()
        self.base_channels = base_channels
        self.time_dim = time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c = base_channels
        self.input = nn.Conv2d(1, c, 3, padding=1)
        self.res1 = ResBlock(c, c, time_dim)
        self.down1 = nn.Conv2d(c, c * 2, 4, stride=2, padding=1)
        self.res2 = ResBlock(c * 2, c * 2, time_dim)
        self.down2 = nn.Conv2d(c * 2, c * 4, 4, stride=2, padding=1)
        self.res3 = ResBlock(c * 4, c * 4, time_dim)

        self.mid1 = ResBlock(c * 4, c * 4, time_dim)
        self.mid2 = ResBlock(c * 4, c * 4, time_dim)

        self.up1 = nn.ConvTranspose2d(c * 4, c * 2, 4, stride=2, padding=1)
        self.res_up1 = ResBlock(c * 4, c * 2, time_dim)
        self.up2 = nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1)
        self.res_up2 = ResBlock(c * 2, c, time_dim)
        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(timestep_embedding(timesteps, self.time_dim))

        h1 = self.res1(self.input(x), t_emb)
        h2 = self.res2(self.down1(h1), t_emb)
        h3 = self.res3(self.down2(h2), t_emb)

        h = self.mid2(self.mid1(h3, t_emb), t_emb)
        h = self.up1(h)
        h = self.res_up1(torch.cat([h, h2], dim=1), t_emb)
        h = self.up2(h)
        h = self.res_up2(torch.cat([h, h1], dim=1), t_emb)
        return self.out(h)


class MNISTClassifier(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(feature_dim, 10)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feat = self.fc(self.features(x))
        logits = self.classifier(feat)
        if return_features:
            return logits, feat
        return logits

