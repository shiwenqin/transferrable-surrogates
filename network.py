import math
from copy import deepcopy

import torch
import torch.nn as nn
from einops.layers.torch import Reduce


def millify(n, bytes=False, return_float=False):
    n = float(n)
    if bytes:
        millnames = ["B", "KB", "MB", "GB", "TB", "PB"]
    else:
        millnames = ["", "K", "M", "B", "T"]
    millidx = max(
        0,
        min(
            len(millnames) - 1,
            int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)),
        ),
    )
    if return_float:
        return n / 10 ** (3 * millidx)
    else:
        return f"{int(n / 10 ** (3 * millidx))}{millnames[millidx]}"


class Network(nn.Module):
    """A network that takes architectural modules and wraps them with a stem and a head."""

    def __init__(self, backbone, backbone_output_shape, output_shape, config):
        super(Network, self).__init__()
        self.config = config
        self.backbone = backbone
        if "einspace" in self.config["search_space"]:
            self.stem = nn.Sequential(
                # conv stem to even number of channels?
                # positional embedding?
                nn.Identity()
            )
        elif self.config["search_space"] == "hnasbench201":
            self.stem = nn.Sequential(
                nn.LazyConv2d(16, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(16),
            )

        if len(backbone_output_shape) == 2:
            self.head = nn.Sequential(
                nn.Linear(backbone_output_shape[1], output_shape),
            )
        elif len(backbone_output_shape) == 3:
            if config["dataset"] in ["darcyflow", "psicov", "cosmic"]:
                self.head = nn.Sequential(
                    # On the wide ResNet backbone, we add an adaptive averaging pooling operation to upsample the features back to their original dimensions before output.
                    nn.AdaptiveAvgPool2d(output_shape),
                    # add in another dimension to match the input
                    nn.Lambda(lambda x: x.unsqueeze(1)),
                )
            else:
                self.head = nn.Sequential(
                    Reduce("b s d -> b s", "mean"),
                    nn.Linear(backbone_output_shape[1], output_shape),
                )
        elif len(backbone_output_shape) == 4:
            if config["dataset"] in ["darcyflow", "psicov", "cosmic"]:
                self.head = nn.Sequential(
                    # On the wide ResNet backbone, we add an adaptive averaging pooling operation to upsample the features back to their original dimensions before output.
                    nn.AdaptiveAvgPool2d(output_shape),
                    # change the channels down to match the input
                    nn.Conv2d(backbone_output_shape[1], 1, kernel_size=1, stride=1, padding=0),
                )
            else:
                self.head = nn.Sequential(
                    Reduce("b c h w -> b c", "mean"),
                    nn.Linear(backbone_output_shape[1], output_shape),
                )
        self.backbone_output_shape = backbone_output_shape
        # print(f"Network")
        # print(self)

    def forward(self, x):
        # print("input to stem", x.shape)
        out = self.stem(x)
        # print("input to backbone", out.shape)
        out = self.backbone(out)
        # print("input to head", out.shape, self.backbone_output_shape)
        out = self.head(out)
        # print("output", out.shape)
        return out

    def forward_window(self, x, L=128, stride=-1):
        _, _, _, s_length = x.shape

        if stride == -1:  # Default to window size
            stride = L
            assert (s_length % L == 0)

        y = torch.zeros_like(x)[:, :1, :, :]
        # print("x, y", x.shape, y.shape)
        counts = torch.zeros_like(x)[:, :1, :, :]
        for i in range((((s_length - L) // stride)) + 1):
            ip = i * stride
            for j in range((((s_length - L) // stride)) + 1):
                jp = j * stride
                out = self.forward(x[:, :, ip:ip + L, jp:jp + L])
                # print("x slice, out", x[:, :, ip:ip + L, jp:jp + L].shape, out.shape)
                # out = out.permute(0, 3, 1, 2).contiguous()
                # print("y slice, out", y[:, :, ip:ip + L, jp:jp + L].shape, out.shape)
                y[:, :, ip:ip + L, jp:jp + L] += out
                counts[:, :, ip:ip + L, jp:jp + L] += torch.ones_like(out)
        return y / counts

    def numel(self):
        num_params = sum([p.numel() for p in self.parameters()])
        return num_params

    def num_parameters(self):
        return f"Num params: {millify(self.numel())}"
