import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from ash import SparseDenseGrid
import pytest


# TODO: now only 3 dim keys is supported for sparse-dense grids
# TBD: better defined and general interpolation
class TestSparseDenseGrid:
    capacity = 100
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

        indices = torch.randint(
            grid.num_cells_per_grid, (100, ), device=self.device
        )
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

        sparse_keys = grid_dim * (
            torch.arange(-1, 2, 1, dtype=torch.int, device=self.device)
            .view(-1, 1)
            .tile((1, in_dim))
        )
        grid.spatial_init_(sparse_keys, dilation=0)

        grid_coords, cell_coords, grid_indices, cell_indices = grid.items()
        coords = grid_coords * grid_dim + cell_coords
        coords = coords.view(-1, in_dim)

        grid_indices, cell_indices, offsets, masks = grid.query(coords)
        assert grid_indices.shape == (len(sparse_keys) * grid_dim**in_dim,)
        assert cell_indices.shape == (len(sparse_keys) * grid_dim**in_dim,)
        assert torch.allclose(offsets, torch.zeros_like(offsets))
        assert masks.all()

        features, masks = grid(coords, interpolation="nearest")
        features.sum().backward(retain_graph=True)
        assert grid.embeddings.grad is not None
        assert masks.all()
        assert torch.allclose(features, grid.embeddings[grid_indices, cell_indices])

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
        pass

    def test_backward(self):
        pass
