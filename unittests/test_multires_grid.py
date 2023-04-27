import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from ash import SparseDenseGrid, SparseDenseGridQuery, SparseDenseGridQueryBackward
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from ash import BoundedMultiResGrid
import pytest


# TODO: now only 3 dim keys is supported for sparse-dense grids
# TBD: better defined and general interpolation
class TestBoundedMultiResGrid:
    capacity = 1000
    device = torch.device("cuda:0")

    def _init_block(self):
        in_dim = 3
        multires_grid = BoundedMultiResGrid(
            in_dim=in_dim,
            num_embeddings=[self.capacity, self.capacity, self.capacity],
            embedding_dims=2,
            grid_dims=2,
            sparse_grid_dims=[2, 4, 8],
            bbox_min=torch.ones(in_dim, device=self.device) * -1,
            bbox_max=torch.ones(in_dim, device=self.device),
            device=self.device,
        )

    def _forward_block(self):
        in_dim = 3
        multires_grid = BoundedMultiResGrid(
            in_dim=in_dim,
            num_embeddings=[self.capacity, self.capacity, self.capacity],
            embedding_dims=2,
            grid_dims=2,
            sparse_grid_dims=[2, 4, 8],
            bbox_min=torch.ones(in_dim, device=self.device) * -1,
            bbox_max=torch.ones(in_dim, device=self.device),
            device=self.device,
        )

        points = torch.rand(100, 3, device=self.device) * 2.0 - 1.0

        multires_grid.spatial_init_(points, dilation=0)

        features = multires_grid(points)
        assert features.shape == (100, 6)

    def _backward_backward_block(self):
        in_dim = 3
        multires_grid = BoundedMultiResGrid(
            in_dim=in_dim,
            num_embeddings=[self.capacity, self.capacity, self.capacity],
            embedding_dims=2,
            grid_dims=2,
            sparse_grid_dims=[2, 4, 8],
            bbox_min=torch.ones(in_dim, device=self.device) * -1,
            bbox_max=torch.ones(in_dim, device=self.device),
            device=self.device,
        )

        points = torch.rand(100, 3, device=self.device, requires_grad=True) * 2.0 - 1.0
        multires_grid.spatial_init_(points, dilation=0)

        features = multires_grid(points)
        assert features.shape == (100, 6)

        pseudo_sdf = torch.matmul(features, torch.rand(6, 1, device=self.device))
        grad_pseudo_sdf = torch.autograd.grad(
            pseudo_sdf[..., 0],
            points,
            torch.ones_like(pseudo_sdf[..., 0], requires_grad=False),
            create_graph=True,
        )[0]
        grad_pseudo_sdf.sum().backward()

    def test_init(self):
        self._init_block()

    def test_forward(self):
        self._forward_block()

    def test_back_backward(self):
        self._backward_backward_block()
