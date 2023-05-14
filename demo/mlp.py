"""
MIT License

Copyright (c) 2022 Yuanchen Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified from
https://github.com/bennyguo/instant-nsr-pl/blob/main/models/network_utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_activation(name):
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name.startswith("scale"):
        scale_factor = float(name[5:])
        return lambda x: x.clamp(0.0, scale_factor) / scale_factor
    elif name.startswith("clamp"):
        clamp_max = float(name[5:])
        return lambda x: x.clamp(0.0, clamp_max)
    elif name.startswith("mul"):
        mul_factor = float(name[3:])
        return lambda x: x * mul_factor
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name.startswith("+") or name.startswith("-"):
        return lambda x: x + float(name)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    else:
        return getattr(F, name)


class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        radius=1.0,
        sphere_init=True,
        weight_norm=True,
        final_activation=None,
    ):
        super().__init__()

        self.dim_hidden = dim_hidden
        self.weight_norm = weight_norm

        self.sphere_init = sphere_init
        self.sphere_init_radius = radius

        self.num_hidden_layers = num_layers - 1
        assert self.num_hidden_layers > 0, "Number of layers must be at least 2"

        self.layers = [
            self.make_linear(dim_in, self.dim_hidden, is_first=True, is_last=False),
            self.make_activation(),
        ]
        for i in range(self.num_hidden_layers - 1):
            self.layers += [
                self.make_linear(
                    self.dim_hidden, self.dim_hidden, is_first=False, is_last=False
                ),
                self.make_activation(),
            ]
        self.layers += [
            self.make_linear(self.dim_hidden, dim_out, is_first=False, is_last=True)
        ]
        self.layers = nn.Sequential(*self.layers)
        self.output_activation = get_activation(final_activation)

    def forward(self, x):
        x = self.layers(x.float())
        x = self.output_activation(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last):
        layer = nn.Linear(dim_in, dim_out, bias=True)
        if self.sphere_init:
            if is_last:
                torch.nn.init.constant_(layer.bias, -self.sphere_init_radius)
                torch.nn.init.normal_(
                    layer.weight,
                    mean=np.sqrt(np.pi) / np.sqrt(dim_in),
                    std=0.0001,
                )
            elif is_first:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.constant_(layer.weight[:, 3:], 0.0)
                torch.nn.init.normal_(
                    layer.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(dim_out)
                )
            else:
                torch.nn.init.constant_(layer.bias, 0.0)
                torch.nn.init.normal_(layer.weight, 0.0, np.sqrt(2) / np.sqrt(dim_out))
        else:
            torch.nn.init.constant_(layer.bias, 0.0)
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")

        if self.weight_norm:
            layer = nn.utils.weight_norm(layer)
        return layer

    def make_activation(self):
        if self.sphere_init:
            return nn.Softplus(beta=100)
        else:
            return nn.ReLU(inplace=True)


if __name__ == "__main__":
    implicit_mlp = MLP(3 + 16, 128, 1 + 13, 3, sphere_init=True, weight_norm=True)
    rgb_mlp = MLP(
        3 + 13,
        128,
        3,
        3,
        sphere_init=False,
        weight_norm=False,
        final_activation="sigmoid",
    )

    x = torch.rand(100, 3)
    d = torch.rand(100, 3)
    pos_feat = torch.rand(100, 16)

    out = implicit_mlp(torch.cat([x, pos_feat], dim=-1))
    sdf, geo_feat = torch.split(out, [1, 13], dim=-1)
    rgb = rgb_mlp(torch.cat([d, geo_feat], dim=-1))

    print(rgb.shape)
