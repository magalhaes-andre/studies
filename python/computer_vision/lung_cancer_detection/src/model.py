from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)

    original_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=1,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if pretrained:
            new_conv.weight.copy_(original_conv.weight.mean(dim=1, keepdim=True))

    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
