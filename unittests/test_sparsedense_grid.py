import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from ash import (
    SparseDenseGrid,
    SparseDenseGridQuery,
    SparseDenseGridQueryBackward,
    BoundedSparseDenseGrid,
)
import pytest


import time
import skimage.measure
import plyfile
import numpy as np
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# helpers

np.random.seed(15213)
torch.manual_seed(15213)

def exists(val):
    return val is not None


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# siren layer


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


# siren network


class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=1.0,
        w0_initial=30.0,
        use_bias=True,
        final_activation=None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(
                Siren(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=layer_w0,
                    use_bias=use_bias,
                    is_first=is_first,
                )
            )

        final_activation = (
            nn.Identity() if not exists(final_activation) else final_activation
        )
        self.last_layer = Siren(
            dim_in=dim_hidden,
            dim_out=dim_out,
            w0=w0,
            use_bias=use_bias,
            activation=final_activation,
        )

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, "d -> () d")

        return self.last_layer(x)




# TODO: now only 3 dim keys is supported for sparse-dense grids
# TBD: better defined and general interpolation
class TestSparseDenseGrid:
    capacity = 1000
    device = torch.device("cuda:0")

    def _init_block(self, in_dim, embedding_dim, grid_dim):
        grid = SparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=self.capacity,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            device=self.device,
        )

        assert grid.embeddings.shape == (
            self.capacity,
            grid_dim**in_dim,
            embedding_dim,
        )

    def _linearize_block(self, in_dim, embedding_dim, grid_dim):
        grid = SparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=self.capacity,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            device=self.device,
        )

        indices = torch.randint(grid.num_cells_per_grid, (100,), device=self.device)
        coords = grid._delinearize_cell_indices(indices)
        indices_linearized = grid._linearize_cell_coords(coords)
        coords_delinearized = grid._delinearize_cell_indices(indices_linearized)

        assert torch.all(indices == indices_linearized)
        assert torch.all(coords == coords_delinearized)

    def _neighbor_block(self, in_dim, embedding_dim, grid_dim):
        grid = SparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=self.capacity,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            device=self.device,
        )

        indices = torch.randint(
            grid.num_cells_per_grid, (100, in_dim), device=self.device
        )
        coords = grid._delinearize_cell_indices(indices)

        nb_indices0 = grid.neighbor_table_cell2cell[indices, 0]
        nb_coords0 = grid._delinearize_cell_indices(nb_indices0)

        assert torch.all(nb_indices0 == indices)
        assert torch.all(nb_coords0 == coords)

        nb_indices_last = grid.neighbor_table_cell2cell[indices, -1]
        nb_coords_last = grid._delinearize_cell_indices(nb_indices_last)
        offset_last = torch.ones_like(nb_coords_last[0])
        nb_coords_last_expected = (coords + offset_last) % grid_dim
        assert torch.all(nb_coords_last == nb_coords_last_expected)

    def _bounded_neighbor_block(self, in_dim, embedding_dim, grid_dim, sparse_grid_dim):
        grid = BoundedSparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=self.capacity,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            sparse_grid_dim=sparse_grid_dim,
            device=self.device,
        )

        grid.dense_init_()




    def _item_block(self, in_dim, embedding_dim, grid_dim):
        grid = SparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=self.capacity,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            device=self.device,
        )

        keys = (
            torch.arange(-1, 1, 1, dtype=torch.long, device=self.device)
            .view(-1, 1)
            .tile((1, in_dim))
        )
        grid.spatial_init_(keys, dilation=0)

        grid_coords, cell_coords, grid_indices, cell_indices = grid.items()

        n = len(keys)
        m = grid_dim**in_dim

        coords = grid_coords * grid_dim + cell_coords
        assert coords.shape == (n, m, in_dim)

        features = grid.embeddings[grid_indices, cell_indices]
        assert features.shape == (n, m, embedding_dim)

    def _query_block(self, in_dim, embedding_dim, grid_dim):
        grid = SparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=self.capacity,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            device=self.device,
        )
        nn.init.uniform_(grid.embeddings, -1, 1)

        grid_coords = grid_dim * (
            torch.arange(-1, 2, 1, dtype=torch.int, device=self.device)
            .view(-1, 1)
            .tile((1, in_dim))
        )
        grid.spatial_init_(grid_coords, dilation=0)

        grid_coords, cell_coords, grid_indices, cell_indices = grid.items()
        coords = grid_coords * grid_dim + cell_coords
        coords = coords.view(-1, in_dim)

        grid_indices, cell_indices, offsets, masks = grid.query(coords)
        assert grid_indices.shape == (len(grid_coords) * grid_dim**in_dim,)
        assert cell_indices.shape == (len(grid_coords) * grid_dim**in_dim,)
        assert torch.allclose(offsets, torch.zeros_like(offsets))
        assert masks.all()

        features, masks = grid(coords, interpolation="nearest")
        features.sum().backward(retain_graph=True)
        assert grid.embeddings.grad is not None
        assert masks.all()
        assert torch.allclose(features, grid.embeddings[grid_indices, cell_indices])

    def _forward_block(self, in_dim, embedding_dim, grid_dim, bound=3):
        grid = SparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=self.capacity,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            device=self.device,
        )

        grid_coord_range = torch.arange(
            -bound, bound + 1, 1, dtype=torch.int, device=self.device
        )

        # Create a dense grid to test correctness
        grid_coords = grid_dim * torch.stack(
            torch.meshgrid(
                grid_coord_range, grid_coord_range, grid_coord_range, indexing="ij"
            ),
            dim=-1,
        ).view(-1, 3)

        grid.spatial_init_(grid_coords, dilation=0)
        grid_coords, cell_coords, grid_indices, cell_indices = grid.items()
        coords = grid_coords * grid_dim + cell_coords
        coords = coords.view(-1, in_dim).float()

        with torch.no_grad():
            grid.embeddings[grid_indices, cell_indices, :3] = coords.view(
                grid_indices.shape[0], cell_indices.shape[1], 3
            )

        # Map query to [min, max - 1) to check purely in-bound queries
        query_cell_coords = torch.rand(10, 3, device=self.device)
        # min: -grid_dim * bound
        # max: grid_dim * bound - 1
        query_cell_coords = (
            2 * grid_dim * bound - 1
        ) * query_cell_coords - grid_dim * bound

        embeddings, masks = grid(query_cell_coords, interpolation="linear")
        assert torch.allclose(embeddings[..., :3], query_cell_coords)
        assert masks.all()

    def _backward_block(self, in_dim, embedding_dim, grid_dim):
        grid = SparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=8,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            device=self.device,
        )

        grid_coord_range = torch.arange(0, 2, 1, dtype=torch.int, device=self.device)
        grid_coords = grid_dim * torch.stack(
            torch.meshgrid(
                grid_coord_range, grid_coord_range, grid_coord_range, indexing="ij"
            ),
            dim=-1,
        ).view(-1, 3)
        grid.spatial_init_(grid_coords, dilation=0)
        torch.nn.init.uniform_(grid.embeddings, -1, 1)

        grid_coords, cell_coords, grid_indices, cell_indices = grid.items()

        # Must make sure that after perturbation, the query is still in the same cell
        # Otherwise things could go wrong
        eps = 1e-3
        num_queries = 3

        query_cell_coords = torch.randint(
            high=grid_dim * 2 - 1, size=(num_queries, in_dim), device=self.device
        )
        query_cell_offsets = (
            torch.rand(num_queries, in_dim, device=self.device)
        ).clamp(2 * eps, 1 - 2 * eps)
        query_positions = query_cell_coords + query_cell_offsets
        query_positions.requires_grad_(True)

        net = SirenNet(embedding_dim + 3, dim_hidden=128, dim_out=1, num_layers=2, w0=30.0).cuda()

        grid.construct_sparse_neighbor_tables_()

        import tinycudann as tcnn
        ngp = tcnn.Encoding(
                3,
                {
                    "otype": "DenseGrid",
                    "n_levels": 1,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 3,
                    "base_resolution": 2,
                    "per_level_scale": 2,
                    "interpolation": "Linear",
                },
                dtype=torch.float32,
            ).to(self.device)

        def grad_ngp_fn(x_input):
            x_input.requires_grad_(True)
            x_input = x_input * 0.5
            output = ngp(x_input)
            print(f"ngp output: {output}")
            output = net(torch.cat((output, x_input), dim=-1))
            print(f"network output: {output}")
            return output

        torch.autograd.gradcheck(
            grad_ngp_fn,
            (query_positions,),
            eps=eps,
            atol=1e-2,
            rtol=1e-2,
        )


        def grad_x_fn(x_input):
            x_input.requires_grad_(True)
            x = grid.transform_world_to_cell(x_input)
            grid_indices, cell_indices, offsets, masks = grid.query(x)
            print(f'[grad_x_fn]')
            print(f'x_input = {x_input}')
            assert masks.all()

            output = SparseDenseGridQuery.apply(
                grid.embeddings,
                offsets,
                grid_indices,
                cell_indices,
                masks,
                grid.neighbor_table_grid2grid,
                grid.neighbor_table_cell2cell,
                grid.neighbor_table_cell2grid,
                grid.grid_dim,
                "linear",
            )
            print(f"grid output: {output}")
            output = net(torch.cat((output, x_input), dim=-1))
            print(f"network output: {output}")
            return output

        torch.autograd.gradcheck(
            grad_x_fn,
            (query_positions,),
            eps=eps,
            atol=1e-2,
            rtol=1e-2,
        )

        # def grad_embedding_fn(embeddings):
        #     x_input = query_positions
        #     x = grid.transform_world_to_cell(x_input)
        #     grid_indices, cell_indices, offsets, masks = grid.query(x)
        #     assert masks.all()

        #     grid.construct_sparse_neighbor_tables_()

        #     output = SparseDenseGridQuery.apply(
        #         embeddings,
        #         offsets,
        #         grid_indices,
        #         cell_indices,
        #         masks,
        #         grid.neighbor_table_grid2grid,
        #         grid.neighbor_table_cell2cell,
        #         grid.neighbor_table_cell2grid,
        #         grid.grid_dim,
        #     )
        #     output = net(torch.cat((output, x_input), dim=-1))
        #     return output

        # for i in range(num_queries):
        #     torch.autograd.gradcheck(
        #         grad_x_fn,
        #         (query_positions[i:i+1],),
        #         eps=eps,
        #         atol=1e-2,
        #         rtol=1e-2,
        #     )





        # torch.autograd.gradcheck(
        #     grad_embedding_fn,
        #     (grid.embeddings,),
        #     eps=eps,
        #     atol=1e-2,
        #     rtol=1e-2,
        # )

    def _bounded_backward_block(self, in_dim, embedding_dim, sparse_grid_dim):
        grid = BoundedSparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=sparse_grid_dim**3,
            embedding_dim=embedding_dim,
            grid_dim=1,
            sparse_grid_dim=sparse_grid_dim,
            device=self.device,
        )
        grid.dense_init_()
        torch.nn.init.uniform_(grid.embeddings, -1, 1)

        # Must make sure that after perturbation, the query is still in the same cell
        # Otherwise things could go wrong
        eps = 1e-3
        num_queries = 1000

        query_cell_coords = torch.randint(
            high=sparse_grid_dim - 1, size=(num_queries, in_dim), device=self.device
        )
        query_cell_offsets = (
            torch.rand(num_queries, in_dim, device=self.device)
        ).clamp(2 * eps, 1 - 2 * eps)
        query_cell_positions = query_cell_coords + query_cell_offsets
        query_positions = grid.transform_cell_to_world(query_cell_positions)
        query_positions.requires_grad_(True)

        def grad_x_fn(x):
            x.requires_grad_(True)
            output, mask = grid(x, interpolation="smooth_step")
            assert mask.all()
            return output

        torch.autograd.gradcheck(
            grad_x_fn,
            (query_positions,),
            eps=eps * grid.cell_size[0],
            atol=1e-3,
            rtol=1e-3,
        )

        # torch.autograd.gradcheck(
        #     gradgrad_x_fn,
        #     (query_positions,),
        #     eps=eps * grid.cell_size[0],
        #     atol=1e-3,
        #     rtol=1e-3,
        # )

    def _backward_backward_block(self, in_dim, embedding_dim, grid_dim):
        grid = SparseDenseGrid(
            in_dim=in_dim,
            num_embeddings=8,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            device=self.device,
        )

        grid_coord_range = torch.arange(0, 2, 1, dtype=torch.int, device=self.device)
        grid_coords = grid_dim * torch.stack(
            torch.meshgrid(
                grid_coord_range, grid_coord_range, grid_coord_range, indexing="ij"
            ),
            dim=-1,
        ).view(-1, 3)
        grid.spatial_init_(grid_coords, dilation=0)
        torch.nn.init.uniform_(grid.embeddings, -1, 1)

        grid_coords, cell_coords, grid_indices, cell_indices = grid.items()

        # Must make sure that after perturbation, the query is still in the same cell
        # Otherwise things could go wrong
        eps = 1e-3
        num_queries = 1000
        query_cell_coords = torch.randint(
            high=grid_dim * 2 - 1, size=(num_queries, in_dim), device=self.device
        )
        query_cell_offsets = (
            torch.rand(num_queries, in_dim, device=self.device)
        ).clamp(2 * eps, 1 - 2 * eps)
        query_positions = query_cell_coords + query_cell_offsets
        query_positions.requires_grad_(True)
        z = torch.rand_like(query_positions, dtype=torch.float64)

        def grad2_embedding_fn(grid_embedding):
            x = grid.transform_world_to_cell(query_positions)

            grid_indices, cell_indices, offsets, masks = grid.query(x)
            assert masks.all()
            grid.construct_sparse_neighbor_tables_()

            grad_embeddings, grad_offsets = SparseDenseGridQueryBackward.apply(
                z,
                grid_embedding,
                offsets,
                grid_indices,
                cell_indices,
                masks,
                grid.neighbor_table_grid2grid,
                grid.neighbor_table_cell2cell,
                grid.neighbor_table_cell2grid,
                grid.grid_dim,
                "linear",
            )
            return grad_embeddings

        # Only check grad_grad w.r.t. embeddings, as backward to offsets is not yet implemented
        torch.autograd.gradcheck(
            grad2_embedding_fn,
            (grid.embeddings.double(),),
            eps=1e-3,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_backward(self):
        self._backward_block(in_dim=3, embedding_dim=2, grid_dim=1)
        return
        self._backward_block(in_dim=3, embedding_dim=1, grid_dim=2)
        self._backward_block(in_dim=3, embedding_dim=4, grid_dim=4)
        self._backward_block(in_dim=3, embedding_dim=8, grid_dim=8)

    def test_init(self):
        self._init_block(in_dim=3, embedding_dim=1, grid_dim=4)
        self._init_block(in_dim=3, embedding_dim=5, grid_dim=8)
        self._init_block(in_dim=3, embedding_dim=2, grid_dim=16)

    def test_linearize(self):
        self._linearize_block(in_dim=3, embedding_dim=1, grid_dim=3)
        self._linearize_block(in_dim=3, embedding_dim=1, grid_dim=4)
        self._linearize_block(in_dim=3, embedding_dim=1, grid_dim=8)
        self._linearize_block(in_dim=3, embedding_dim=1, grid_dim=16)

    def test_neighbor(self):
        self._neighbor_block(in_dim=3, embedding_dim=1, grid_dim=2)
        self._neighbor_block(in_dim=3, embedding_dim=1, grid_dim=2)
        self._neighbor_block(in_dim=3, embedding_dim=8, grid_dim=4)
        self._neighbor_block(in_dim=3, embedding_dim=16, grid_dim=8)

    def test_items(self):
        self._item_block(in_dim=3, embedding_dim=1, grid_dim=2)
        self._item_block(in_dim=3, embedding_dim=1, grid_dim=2)
        self._item_block(in_dim=3, embedding_dim=8, grid_dim=4)
        self._item_block(in_dim=3, embedding_dim=16, grid_dim=8)

    def test_query(self):
        self._query_block(in_dim=3, embedding_dim=1, grid_dim=2)
        self._query_block(in_dim=3, embedding_dim=1, grid_dim=8)
        self._query_block(in_dim=3, embedding_dim=5, grid_dim=8)
        self._query_block(in_dim=3, embedding_dim=16, grid_dim=8)

    def test_forward(self):
        self._forward_block(in_dim=3, embedding_dim=3, grid_dim=5)


    # def test_bounded_backward(self):
    #     self._bounded_backward_block(in_dim=3, embedding_dim=1, sparse_grid_dim=8)
    #     self._bounded_backward_block(in_dim=3, embedding_dim=8, sparse_grid_dim=8)

    # def test_backward_backward(self):
    #     self._backward_backward_block(in_dim=3, embedding_dim=1, grid_dim=1)
    #     self._backward_backward_block(in_dim=3, embedding_dim=3, grid_dim=2)
