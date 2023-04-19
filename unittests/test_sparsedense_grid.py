import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from ash import SparseDenseGrid, SparseDenseGridQuery, SparseDenseGridQueryBackward
import pytest


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
        num_queries = 1000

        query_cell_coords = torch.randint(
            high=grid_dim * 2 - 1, size=(num_queries, in_dim), device=self.device
        )
        query_cell_offsets = (
            torch.rand(num_queries, in_dim, device=self.device)
        ).clamp(2 * eps, 1 - 2 * eps)
        query_positions = query_cell_coords + query_cell_offsets
        query_positions.requires_grad_(True)

        def grad_x_fn(x):
            x = grid.transform_world_to_cell(x)
            grid_indices, cell_indices, offsets, masks = grid.query(x)
            assert masks.all()

            grid.construct_sparse_neighbor_tables_()

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
            )
            return output

        def grad_embedding_fn(embeddings):
            x = grid.transform_world_to_cell(query_positions)
            grid_indices, cell_indices, offsets, masks = grid.query(x)
            assert masks.all()

            grid.construct_sparse_neighbor_tables_()

            output = SparseDenseGridQuery.apply(
                embeddings,
                offsets,
                grid_indices,
                cell_indices,
                masks,
                grid.neighbor_table_grid2grid,
                grid.neighbor_table_cell2cell,
                grid.neighbor_table_cell2grid,
                grid.grid_dim,
            )
            return output

        torch.autograd.gradcheck(
            grad_x_fn,
            (query_positions,),
            eps=eps,
            atol=1e-3,
            rtol=1e-3,
        )
        torch.autograd.gradcheck(
            grad_embedding_fn,
            (grid.embeddings,),
            eps=eps,
            atol=1e-3,
            rtol=1e-3,
        )

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

    def test_backward(self):
        self._backward_block(in_dim=3, embedding_dim=1, grid_dim=1)
        self._backward_block(in_dim=3, embedding_dim=1, grid_dim=2)
        self._backward_block(in_dim=3, embedding_dim=1, grid_dim=4)
        self._backward_block(in_dim=3, embedding_dim=3, grid_dim=8)

    def test_backward_backward(self):
        self._backward_backward_block(in_dim=3, embedding_dim=1, grid_dim=1)
        self._backward_backward_block(in_dim=3, embedding_dim=3, grid_dim=2)
