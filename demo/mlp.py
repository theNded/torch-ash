
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mesh_util import create_mesh

from ash import MLP

if __name__ == "__main__":
    implicit_mlp = MLP(
        3 + 16, 128, 1 + 13, 3, radius=0.8, sphere_init=True, weight_norm=True
    ).cuda()
    rgb_mlp = MLP(
        3 + 13,
        128,
        3,
        3,
        sphere_init=False,
        weight_norm=False,
        final_activation="sigmoid",
    ).cuda()

    x = torch.rand(100, 3).cuda()
    d = torch.rand(100, 3).cuda()
    pos_feat = torch.rand(100, 16).cuda()

    out = implicit_mlp(torch.cat([x, pos_feat], dim=-1))
    sdf, geo_feat = torch.split(out, [1, 13], dim=-1)
    rgb = rgb_mlp(torch.cat([d, geo_feat], dim=-1))

    def sdf_fn(x):
        pos_feat = torch.rand((x.shape[0], 16)).cuda()
        out = implicit_mlp(torch.cat([x, pos_feat], dim=-1))
        sdf = out[:, 0]
        return sdf.detach()

    create_mesh(sdf_fn, "test.ply")
