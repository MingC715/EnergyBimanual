# Adapted from https://github.com/openai/CLIP/blob/main/clip/model.py

import torch
from torch import nn

from .clip_resnet import ModifiedResNet


class CLIPTransform(nn.Module):

    def __init__(self):
        super().__init__()
        _mean = [0.48145466, 0.4578275, 0.40821073]
        _std = [0.26862954, 0.26130258, 0.27577711]
        self.register_buffer("mean", torch.tensor(_mean).reshape(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(_std).reshape(1, -1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


def load_clip():
    # The released state dict contains the full RN50 visual encoder.
    backbone = ModifiedResNetFeatures((3, 4, 6, 3), output_dim=1024, heads=32)
    normalize = CLIPTransform()
    return backbone, normalize


class ModifiedResNetFeatures(ModifiedResNet):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__(layers, output_dim, heads, input_resolution, width)

    def forward(self, x: torch.Tensor):
        x = x.type(self.conv1.weight.dtype)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x0 = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return {
            "res1": x0,
            "res2": x1,
            "res3": x2,
            "res4": x3,
            "res5": x4,
        }
