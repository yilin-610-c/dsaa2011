import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class PlainCNN(nn.Module):
    def __init__(self, num_classes: int = 10, width_multiplier: float = 1.0) -> None:
        super().__init__()
        c1 = int(32 * width_multiplier)
        c2 = int(64 * width_multiplier)
        c3 = int(128 * width_multiplier)
        c4 = int(256 * width_multiplier)
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(c4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ShortcutA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x[:, :, :: self.stride, :: self.stride]
        channel_gap = self.out_channels - self.in_channels
        pad_left = channel_gap // 2
        pad_right = channel_gap - pad_left
        return F.pad(identity, (0, 0, 0, 0, pad_left, pad_right), "constant", 0)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        shortcut_type: str = "B",
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()

        if stride != 1 or in_channels != out_channels:
            if shortcut_type == "A":
                self.shortcut = ShortcutA(in_channels, out_channels, stride)
            else:
                self.shortcut = nn.Sequential(
                    conv1x1(in_channels, out_channels, stride),
                    nn.BatchNorm2d(out_channels),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = conv1x1(in_channels, out_channels, stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        preact = self.relu1(self.bn1(x))
        identity = self.shortcut(preact if not isinstance(self.shortcut, nn.Identity) else x)
        out = self.conv1(preact)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = out + identity
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Callable[..., nn.Module],
        layers: List[int],
        num_classes: int = 10,
        width_multiplier: float = 1.0,
        shortcut_type: str = "B",
    ) -> None:
        super().__init__()
        base_channels = [64, 128, 256, 512]
        channels = [int(c * width_multiplier) for c in base_channels]
        self.in_channels = channels[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1, shortcut_type=shortcut_type)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2, shortcut_type=shortcut_type)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2, shortcut_type=shortcut_type)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2, shortcut_type=shortcut_type)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[3] * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(
        self,
        block: Callable[..., nn.Module],
        out_channels: int,
        blocks: int,
        stride: int,
        shortcut_type: str,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers: List[nn.Module] = []
        for block_stride in strides:
            if block is PreActBlock:
                layer = block(self.in_channels, out_channels, block_stride)
            else:
                layer = block(self.in_channels, out_channels, block_stride, shortcut_type=shortcut_type)
            layers.append(layer)
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    residual: bool
    shortcut_type: str
    block_type: str
    depth: int
    width_multiplier: float
    stem: str
    downsample_strategy: str


def build_model(model_name: str, num_classes: int = 10) -> Tuple[nn.Module, ModelSpec]:
    if model_name == "plain_cnn":
        model = PlainCNN(num_classes=num_classes)
        spec = ModelSpec(
            model_id="M1",
            family="Plain-CNN",
            residual=False,
            shortcut_type="N/A",
            block_type="PlainConvBlock",
            depth=0,
            width_multiplier=1.0,
            stem="3x3 conv",
            downsample_strategy="stride",
        )
        return model, spec

    if model_name == "resnet18_optb":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_multiplier=1.0, shortcut_type="B")
        spec = ModelSpec(
            model_id="M2",
            family="ResNet-18-OptB",
            residual=True,
            shortcut_type="OptionB(1x1 proj)",
            block_type="BasicBlock",
            depth=18,
            width_multiplier=1.0,
            stem="3x3 conv",
            downsample_strategy="stage stride=2",
        )
        return model, spec

    if model_name == "resnet18_opta":
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width_multiplier=1.0, shortcut_type="A")
        spec = ModelSpec(
            model_id="M3",
            family="ResNet-18-OptA",
            residual=True,
            shortcut_type="OptionA(zero-pad)",
            block_type="BasicBlock",
            depth=18,
            width_multiplier=1.0,
            stem="3x3 conv",
            downsample_strategy="stage stride=2",
        )
        return model, spec

    if model_name == "preact_resnet18":
        model = ResNet(PreActBlock, [2, 2, 2, 2], num_classes=num_classes, width_multiplier=1.0, shortcut_type="B")
        spec = ModelSpec(
            model_id="M4",
            family="PreAct-ResNet-18",
            residual=True,
            shortcut_type="OptionB(1x1 proj)",
            block_type="PreActBasicBlock",
            depth=18,
            width_multiplier=1.0,
            stem="3x3 conv",
            downsample_strategy="stage stride=2",
        )
        return model, spec

    if model_name == "wide_resnet14":
        model = ResNet(BasicBlock, [1, 2, 2, 1], num_classes=num_classes, width_multiplier=2.0, shortcut_type="B")
        spec = ModelSpec(
            model_id="M5",
            family="Wide-ResNet-14",
            residual=True,
            shortcut_type="OptionB(1x1 proj)",
            block_type="BasicBlock",
            depth=14,
            width_multiplier=2.0,
            stem="3x3 conv",
            downsample_strategy="stage stride=2",
        )
        return model, spec

    raise ValueError(f"Unsupported model name: {model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

