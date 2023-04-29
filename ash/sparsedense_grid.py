from typing import List, Union, Tuple, Dict, OrderedDict, Optional, Literal, overload

import torch
import torch.nn as nn
from .core import ASHEngine, ASHModule
from .hashmap import HashMap, HashSet

from .common import _get_c_extension

backend = _get_c_extension()

"""
Glossary:
SparseDenseGrid is constructed by:
    - sparse [grids], each of which is a dense array of
    - [cells], whose element number is grid_dim**3

It is a single-resolution grid but allocated by geometry spatial distribution.
Each cell's embedding is stored separately, and can be accessed by
    embeddings[grid_idx, cell_idx].
"""


def enumerate_neighbors(dim: int, radius: int, bidirectional: bool) -> torch.Tensor:
    """Generate neighbor coordinate offsets.
    In the 1-radius, non-bidirectional case, it is equivalent to morton code
    i.e., 001, 010, 011, 100, 101, 110, 111 (3 digits in zyx order)
    Args:
        dim: dimension of the coordinate
        radius: radius of the neighborhood
        bidirectional: whether to include negative offsets.
    Returns:
        If bidirectional: enumerating [-r, -r, -r] -- [r, r, r]
          returns a tensor of shape ((2 * radius + 1) ** dim, dim)
        If not bidirectional: enumerating [0, 0, 0] -- [r, r, r]
          returns a tensor of shape ((radius + 1) ** dim, dim)
    """
    if bidirectional:
        arange = torch.arange(-radius, radius + 1)
    else:
        arange = torch.arange(0, radius + 1)

    offsets = (
        torch.stack(torch.meshgrid(*[arange for _ in range(dim)], indexing="ij"))
        .reshape(dim, -1)
        .T.flip(dims=(1,))  # zyx order => xyz for easier adding
        .contiguous()
    )

    return offsets


class SparseDenseGridQuery(torch.autograd.Function):
    """Interpolate the embeddings.
    Each query point x can be located in a cell with grid_indices and cell_indices.
    The offset defined by its relative position to the cell corner gives interpolation ratio in the cell unit.

    Therefore the output is differentiable w.r.t. embeddings and offsets.
    ----------
    |     |  |
    |---->x  |
    |        |
    ----------
    """

    @staticmethod
    def forward(
        ctx,
        embeddings: torch.Tensor,
        offsets: torch.Tensor,
        grid_indices: torch.Tensor,
        cell_indices: torch.Tensor,
        masks: torch.Tensor,
        neighbor_table_grid2grid: torch.Tensor,
        neighbor_table_cell2cell: torch.Tensor,
        neighbor_table_cell2grid: torch.Tensor,
        grid_dim: int,
        interpolation: Literal["linear", "smooth_step"] = "smooth_step",
    ) -> torch.Tensor:
        """
        Forward pass of the interpolation.
        For simplicity, we only consider a single query point offset of (3,) and its 8 neighbors
        y = \\sum_{i=0}^7 weight(offset)[i] * embeddings[i]
        Args:
            embeddings: (num_embeddings, cells_per_grid, embedding_dim) embeddings of the grid [differentiable]
            offsets: (num_queries, 3) offsets of the input [differentiable]

            grid_indices: (num_queries, 1) grid index of the input
            cell_indices: (num_queries, 1) cell index of the input
            masks: (num_queries, 1) mask of the input

            neighbor_table_grid2grid: (num_embeddings, 8) precomputed neighbor table from grid index to grid index
            neighbor_table_cell2cell: (cells_per_grid, 8) precomputed neighbor table from cell index to cell index
            neighbor_table_cell2grid: (cells_per_grid, 8) precomputed neighbor table from cell index to grid index

            grid_dim: int cells_per_grid = grid_dim**3

        Returns:
            y: (num_queries, embedding_dim) interpolated embeddings
        """
        ctx.save_for_backward(
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            neighbor_table_grid2grid,
            neighbor_table_cell2cell,
            neighbor_table_cell2grid,
        )

        ctx.grid_dim = grid_dim
        ctx.interpolation = interpolation

        y = backend.query_forward(
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            neighbor_table_grid2grid,
            neighbor_table_cell2cell,
            neighbor_table_cell2grid,
            grid_dim,
            interpolation,
        )
        return y

    @staticmethod
    def backward(ctx, z: torch.Tensor):
        """Backward pass of the interpolation.
        Supports both forward (for explicit gradient computation via autograd.grad)
        and the conventional backward.
        Detailed in SparseDenseGridQueryBackward.
        """
        (
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            neighbor_table_grid2grid,
            neighbor_table_cell2cell,
            neighbor_table_cell2grid,
        ) = ctx.saved_tensors

        grad_embeddings, grad_offsets = SparseDenseGridQueryBackward.apply(
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            neighbor_table_grid2grid,
            neighbor_table_cell2cell,
            neighbor_table_cell2grid,
            ctx.grid_dim,
            ctx.interpolation,
        )
        return (
            grad_embeddings,
            grad_offsets,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SparseDenseGridQueryBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z,
        embeddings,
        offsets,
        grid_indices,
        cell_indices,
        masks,
        neighbor_table_grid2grid,
        neighbor_table_cell2cell,
        neighbor_table_cell2grid,
        grid_dim,
        interpolation,
    ):
        """
        Forward pass of the backward function.
        Args:
            z: (num_queries, embedding_dim) gradient of the output w.r.t. y
            z could be the gradient of the loss w.r.t. y, i.e.,
                output = dL/dy * dy/dembeddings, dL/dy * dy/doffsets
                       = dL/dembeddings, dL/doffsets
            z could also be a all-one tensor to get jvp(x), i.e.,
                output = dy/dembeddings, dy/doffsets

            Since y = \\sum_{i=0}^7 weight(offset)[i] * embeddings[i]

            grad_embeddings[i] = z * weight(offset)[i]
            grad_outputs = (z * embeddings[i]) * grad_weight(offset)[i]

        weight = [(1-x)(1-y)(1-z), x(1-y)(1-z), ..., xyz] (1 x 8)
        grad_weight = [[-(1-y)(1-z), (1-y)(1-z), ..., yz] (3 x 8)
                       [-(1-x)(1-z), -x(1-z), ...,    xz]
                       [-(1-x)(1-y), -x(1-y), ...,    xy]
        hessian_weight = [[0 1-z 1-y]            [0 z y]  ((3,3) x 8)
                          [1-z 0 1-x]            [z 0 x]
                          [1-y 1-x 0], ...,      [y x 0]]

        Returns:
            grad_embeddings: (num_embeddings, cells_per_grid, embedding_dim) gradient of the embeddings
            grad_offsets: (num_queries, 3)
        """
        ctx.save_for_backward(
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            neighbor_table_grid2grid,
            neighbor_table_cell2cell,
            neighbor_table_cell2grid,
        )
        ctx.grid_dim = grid_dim
        ctx.interpolation = interpolation

        grad_embeddings, grad_offsets = backend.query_backward_forward(
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            neighbor_table_grid2grid,
            neighbor_table_cell2cell,
            neighbor_table_cell2grid,
            grid_dim,
            interpolation,
        )

        return grad_embeddings, grad_offsets

    @staticmethod
    def backward(
        ctx, grad_grad_embeddings: torch.Tensor, grad_grad_offsets: torch.Tensor
    ):
        """Backward pass of the backward function.
        When a gradient is computed in by the backward's forward pass and used to compute a loss, its gradient
        need to be properly back propagated back to embeddings and offsets.
        Args:
            z: (num_queries, embedding_dim) gradient of the output w.r.t. y
            grad_grad_embeddings: (num_embeddings, cells_per_grid, embedding_dim) gradient of the embeddings
            grad_grad_offsets: (num_queries, 3)

            Let
                w1 = grad_embeddings[i] = z * weight(offset)[i], grad_w1 = grad_grad_embeddings[i]
                w2 = grad_offsets = (z * embeddings[i]) * grad_weight(offset)[i], grad_w2 = grad_grad_offsets
            We know that w1 is not used in the forward pass, so we can safely ignore dL/dw1.
            Then we have
                grad_embeddings[i] = grad_w2 * dw1/dembeddings[i]
                                   = (grad_w2  * grad_weight(offset)[i]) * z => (1, num_embeddings)
                grad_offsets = grad_w2 * dw2/doffsets
                             = (z * embeddings[i]) * grad_w2 * hessian_weight(offset)[i] => (1, 3)
        TODO: at current, grad_offsets are skipped as offsets are not optimized.

        Returns:
            grad_embeddings: (num_embeddings, cells_per_grid, embedding_dim) gradient of the embeddings
            grad_offsets: (num_queries, 3)
        """

        (
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            neighbor_table_grid2grid,
            neighbor_table_cell2cell,
            neighbor_table_cell2grid,
        ) = ctx.saved_tensors

        # Safely ignore dL_(dLdembedding) as dLdembedding is not used in the forward pass
        grad_embeddings, grad_offsets = backend.query_backward_backward(
            grad_grad_embeddings,
            grad_grad_offsets,
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            neighbor_table_grid2grid,
            neighbor_table_cell2cell,
            neighbor_table_cell2grid,
            ctx.grid_dim,
            ctx.interpolation,
        )
        return (
            None,
            grad_embeddings,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SparseDenseGrid(ASHModule):
    """
    Create an embedding of (num_embeddings, grid_dim^in_dim * embedding_dim).
    The embedding is a sparse-dense grid, where each [grid] is a dense [array of cells].

    The unit of input is measured by the number of cells.
    Here is the coordinate convention of a sparse-dense grid with grid_dim = 3.
    (Note the local grids are dense, the indices are only selectively shown for clarity.)

    In this case, coordinate (1.2, 1.4) will be mapped to cell X (valid),
    coordinate (0.2, -0.5) will be mapped to cell Y (empty, invalid).
     ___ ___ ___
    |-3,|   |-3,|
    |-3 |___|-1_|
    |   |-2,|   |
    |___|-2 |___|
    |   |   |-1,|
    |___|___|-1_|___ ___ ___ ___ ___ ___
                |0,0|   |   |0,3|   |   |
              Y |___|___|___|___|___|___|
                |   |1,1|   |   |1,4|   |
                |___|_X_|___|___|___|___|
                |2,0|   |2,2|   |   |2,5|
                |___|___|___|___|___|___|
                            |3,3|   |   |
                            |___|___|___|
                            |   |4,4|   |
                            |___|___|___|
                            |   |   |5,5|
                            |___|___|___|
    """

    def __init__(
        self,
        in_dim: int,
        num_embeddings: int,
        embedding_dim: int,
        grid_dim: int,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ):
        assert in_dim == 3, "Only 3D sparse-dense grid is supported for now."
        assert embedding_dim <= 16, "Embedding dim must be <= 16 for now."

        super().__init__()

        self.transform_world_to_cell = lambda x: x
        self.transform_cell_to_world = lambda x: x

        self.in_dim = in_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.grid_dim = grid_dim
        self.device = isinstance(device, str) and torch.device(device) or device

        self.engine = ASHEngine(in_dim, num_embeddings, device)
        self.num_cells_per_grid = grid_dim**in_dim
        self.embeddings = nn.Parameter(
            torch.zeros(
                num_embeddings, self.num_cells_per_grid, embedding_dim, device=device
            )
        )

        # 8 neighbors for trilinear interpolation reshaped for boradcasting
        self.neighbor_coord_offsets = (
            enumerate_neighbors(in_dim, 1, False).to(self.device).view(1, -1, in_dim)
        )

        # Dense grid look up tables for each cell and their neighbors
        self.cell_indices = torch.arange(
            self.num_cells_per_grid, dtype=torch.long, device=self.device
        )
        self.cell_coords = self._delinearize_cell_indices(self.cell_indices)
        assert self.cell_coords.shape == (self.num_cells_per_grid, in_dim)

        dense_neighbor_coords = (
            self.cell_coords.view(-1, 1, in_dim) + self.neighbor_coord_offsets
        ).view(-1, 3)

        cell_boundary_mask = (dense_neighbor_coords == grid_dim).long()
        self.neighbor_table_cell2grid = torch.zeros(
            self.num_cells_per_grid * 2**in_dim, device=self.device, dtype=torch.long
        )
        for i in range(in_dim):
            self.neighbor_table_cell2grid = (
                self.neighbor_table_cell2grid + cell_boundary_mask[:, i] * (2**i)
            )

        self.neighbor_table_cell2cell = self._linearize_cell_coords(
            dense_neighbor_coords % grid_dim
        ).view(self.num_cells_per_grid, -1)

        assert self.neighbor_table_cell2cell.shape == (
            self.num_cells_per_grid,
            2**in_dim,
        )

        # Sparse grid look up tables for each grid and their neighbors
        # Need to be constructed after spatial intialization
        self.grid_coords = None
        self.neighbor_table_grid2grid = None

    @torch.no_grad()
    def _linearize_cell_coords(self, cell_coords: torch.Tensor) -> torch.Tensor:
        assert len(cell_coords.shape) == 2 and cell_coords.shape[1] == self.in_dim

        """Convert dense coordinates to dense indices."""
        cell_indices = torch.zeros_like(
            cell_coords[:, 0], dtype=torch.long, device=self.device
        )
        for i in range(self.in_dim):
            cell_indices += cell_coords[:, i] * self.grid_dim**i
        return cell_indices.long()

    @torch.no_grad()
    def _delinearize_cell_indices(self, cell_indices: torch.Tensor) -> torch.Tensor:
        """Convert dense indices to dense coordinates."""
        cell_coords = []
        cell_indices_iter = cell_indices.clone()
        for i in range(self.in_dim):
            cell_coords.append(cell_indices_iter % self.grid_dim)
            cell_indices_iter = torch.div(
                cell_indices_iter, self.grid_dim, rounding_mode="floor"
            )
        cell_coords = torch.stack(cell_coords, dim=1)
        return cell_coords.int()

    @torch.no_grad()
    def construct_sparse_neighbor_tables_(self, radius=1, bidirection=False) -> None:
        """Construct a neighbor lookup table for the sparse-dense grid.
        This should only be called once the sparse grid is initialized.
        Used for trilinear interpolation in query and marching cubes.
        Updates:
            self.grid_coords: (num_embeddings, in_dim)
            self.neighbor_table_grid2grid: (num_embeddings, 2**in_dim)
        For non-active entries, the neighbor indices are set to -1.
        """
        active_grid_coords, active_grid_indices = self.engine.items()

        active_sparse_neighbor_coords = (
            active_grid_coords.view(-1, 1, self.in_dim).int()
            + self.neighbor_coord_offsets.view(1, -1, self.in_dim).int()
        ).view(-1, self.in_dim)

        # (N*2**in_dim, ), (N*2**in_dim, )
        active_sparse_neighbor_indices, active_sparse_neighbor_masks = self.engine.find(
            active_sparse_neighbor_coords
        )
        # Set not found neighbor indices to -1
        active_sparse_neighbor_indices[~active_sparse_neighbor_masks] = -1

        # Create a dense lookup table for sparse coords and neighbor indices
        self.grid_coords = torch.empty(
            (self.num_embeddings, self.in_dim), dtype=torch.int32, device=self.device
        )
        self.grid_coords.fill_(-1)
        self.grid_coords[active_grid_indices] = active_grid_coords

        self.neighbor_table_grid2grid = torch.empty(
            self.num_embeddings, 2**self.in_dim, dtype=torch.int64, device=self.device
        )
        self.neighbor_table_grid2grid.fill_(-1)
        self.neighbor_table_grid2grid[
            active_grid_indices
        ] = active_sparse_neighbor_indices.view(-1, 8)

    def query(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns:
        grid_indices: (N, 1) tensor of sparse grid indices
        cell_indices: (N, 1) tensor of dense grid indices
        offsets: (N, in_dim) tensor of offsets in dense grid units
        """
        # TODO: merge in one pass for efficiency if necessary
        x_sparse = (x / self.grid_dim).floor()
        x_dense = x - x_sparse * self.grid_dim
        offsets = x_dense - x_dense.floor()

        grid_indices, masks = self.engine.find(x_sparse.int())
        cell_indices = self._linearize_cell_coords(x_dense.int())

        return grid_indices, cell_indices, offsets, masks

    def forward(
        self,
        x: torch.Tensor,
        interpolation: Literal["nearest", "linear", "smooth_step"] = "smooth_step",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Used for differentiable query at floating-point locations.
        Args:
            x: (N, in_dim) tensor of keys, converted to the unit of dense grid cells.
        Returns:
            values: (N, embedding_dim) tensor of features
            masks: (N, 1) tensor of masks
        """
        x = self.transform_world_to_cell(x)
        # print(x.min(), x.max())

        grid_indices, cell_indices, offsets, masks = self.query(x)
        # print(grid_indices, cell_indices, offsets, masks)

        if interpolation == "nearest":
            return self.embeddings[grid_indices, cell_indices], masks

        elif interpolation in ["linear", "smooth_step"]:
            if self.grid_coords is None or self.neighbor_table_grid2grid is None:
                self.construct_sparse_neighbor_tables_()

            features = SparseDenseGridQuery.apply(
                self.embeddings,
                offsets,
                grid_indices,
                cell_indices,
                masks,
                self.neighbor_table_grid2grid,
                self.neighbor_table_cell2cell,
                self.neighbor_table_cell2grid,
                self.grid_dim,
                interpolation,
            )
            return features, masks
        else:
            raise ValueError(f"Unknown interpolation: {interpolation}")

    def items(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used for direct assignment, e.g. initialization with fusion, marching cubes.
        Queries happen at integer locations.

        Here M = grid_dim**in_dim (items per dense grid)
        Returns:
            grid_coords: (N, 1, in_dim) tensor of keys
            cell_coords: (1, M, in_dim) tensor of keys
            grid_indices: (N, 1) tensor of features
            cell_indices: (1, M) tensor of features

        For compactness, the coordinates are viewed with the
        (sparse-dim, dense-dim, [embedding-dim]) convention for easier broadcasting.

        The coordinates in the sparse-dense system can be computed by
            x = grid_coords * grid_dim + cell_coords
        Their associated embedding can be accessed by
            feats = self.embeddings[grid_indices, cell_indices]
        """
        grid_coords, grid_indices = self.engine.items()

        return (
            grid_coords.view(-1, 1, self.in_dim),
            self.cell_coords.view(1, -1, self.in_dim),
            grid_indices.view(-1, 1),
            self.cell_indices.view(1, -1),
        )

    def cell_to_world(
        self, grid_coords: torch.Tensor, cell_coords: torch.Tensor
    ) -> torch.Tensor:
        """Converts cell coordinates to world coordinates.
        Args:
            grid_coords: (N, 1, in_dim) tensor of sparse coordinates
            cell_coords: (1, M, in_dim) tensor of dense coordinates
        Returns:
            world_coords: (N, M, in_dim) tensor of world coordinates
        """
        return self.transform_cell_to_world(
            grid_coords.view(-1, 1, self.in_dim) * self.grid_dim
            + cell_coords.view(1, -1, self.in_dim)
        ).view(-1, self.in_dim)

    def grids_in_bound(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """Placeholder for bounded version"""
        return grid_coords

    def spatial_init_(self, points: torch.Tensor, dilation: int = 1) -> None:
        """Initialize the grid with points in the world coordinate system.
        Args:
            keys: (N, in_dim) tensor of points in the world coordinate system
            dilation: dilation radius per cell of input
        Returns:
            Affected sparse/dense coordinates and indices
        """
        # TODO(wei): optimize for speed if necessary
        points = self.transform_world_to_cell(points)

        grid_coords = torch.floor(points / self.grid_dim).int()
        grid_coords = self.grids_in_bound(grid_coords)

        neighbor_coord_offsets = enumerate_neighbors(
            self.in_dim, dilation, bidirectional=False
        ).to(self.device)
        grid_coords_with_neighbors = (
            grid_coords.view(-1, 1, 3) + neighbor_coord_offsets
        ).view(-1, 3)

        # No need to use a huge hash set due to the duplicates
        hash_set = HashSet(key_dim=3, capacity=len(grid_coords), device=self.device)
        hash_set.insert(grid_coords_with_neighbors)
        unique_grid_coords = hash_set.keys()
        unique_grid_coords = self.grids_in_bound(unique_grid_coords)

        self.engine.insert_keys(unique_grid_coords)
        grid_indices, masks = self.engine.find(unique_grid_coords)
        assert masks.all()

        return (
            unique_grid_coords.view(-1, 1, self.in_dim),
            self.cell_coords.view(1, -1, self.in_dim),
            grid_indices.view(-1, 1),
            self.cell_indices.view(1, -1),
        )

    # Geometry-based initialization
    def ray_init_(
        self,
        ray_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_near: torch.Tensor,
        rays_far: torch.Tensor,
        dilation: int = 3,
    ) -> None:
        """Initialize the grid with rays.
        Args:
            ray_o: (N, in_dim) tensor of ray origins
            rays_d: (N, in_dim) tensor of ray directions
            rays_near: (N, 1) tensor of ray near range
            rays_far: (N, 1) tensor of ray far range
            dilation: dilation factor per cell along the ray
        """
        pass

    # Sampling
    def ray_sample(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_near: torch.Tensor,
        rays_far: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the sparse-dense grid along rays.
        Args:
            rays_o: (N, in_dim) tensor of ray origins
            rays_d: (N, in_dim) tensor of ray directions
            rays_near: (N, 1) tensor of ray near range
            rays_far: (N, 1) tensor of ray far range
        Following nerfacc convention to reuse its rendering
        Returns:
            ray_indices: (M, 1) associated ray index per sample
            t_near: (M, 1) near range of the interval per sample
            t_far: (M, 1) far range of the interval per sample
        Samples can be obtained by
            t = (t_near + t_far) / 2
            x = rays_o[ray_indices] + t * rays_d[ray_indices]
        Rendering could be done by a renderer that interprets ray_indices
        for scattered sum.
        """
        pass

    def uniform_sample(
        self, num_samples: int, space: Literal["occupied", "empty"] = "occupied"
    ) -> torch.Tensor:
        """Sample the sparse-dense grid uniformly.
        Args:
            num_samples: number of samples
            space: "occupied" or "empty" space
        Returns:
            samples: (num_samples, in_dim) tensor of samples
        """
        pass

    def marching_cubes(
        self,
        tsdfs: torch.Tensor,
        weights: torch.Tensor,
        color_fn: None,
        normal_fn: None,
        iso_value: float = 0.0,
        weight_thr: float = 1.0,
        vertices_only=False,
    ) -> torch.Tensor:
        import open3d as o3d

        """Extract isosurface from the grid.
        Args:
            tsdfs: (num_embeddings, num_cells_per_grid) tensor of tsdfs
            weights: (num_embeddings, num_cells_per_grid) tensor of weights
            grid_coords: (N, 3) tensor of sparse coordinates
            grid_indices: (N, 1) tensor of sparse indices, N <= num_embeddings
            sparse_neighbor_indices: (N, 2**3) tensor of sparse neighbor indices. -1 means no neighbor
            iso_value: iso value to extract surface
            weight_thr: weight threshold to consider a cell as occupied
        """

        if self.grid_coords is None or self.neighbor_table_grid2grid is None:
            self.construct_sparse_neighbor_tables_()

        # Get active entries
        grid_coords, grid_indices = self.engine.items()

        if vertices_only:
            vertices = backend.isosurface_extraction(
                tsdfs,
                weights,
                grid_indices,
                self.grid_coords,
                self.neighbor_table_grid2grid,
                self.cell_coords,
                self.neighbor_table_cell2cell,
                self.neighbor_table_cell2grid,
                self.grid_dim,
                0.0,
                1.0,
            )
            positions = self.transform_cell_to_world(vertices)
            if len(positions) == 0:
                return None

            pcd = o3d.t.geometry.PointCloud(positions.detach().cpu().numpy())
            if color_fn is not None:
                colors = color_fn(positions)
                pcd.point["colors"] = colors.detach().cpu().numpy()
            if normal_fn is not None:
                normals = normal_fn(positions)
                pcd.point["normals"] = normals.detach().cpu().numpy()
            return pcd

        else:
            triangles, vertices = backend.marching_cubes(
                tsdfs,
                weights,
                grid_indices,
                self.grid_coords,
                self.neighbor_table_grid2grid,
                self.cell_coords,
                self.neighbor_table_cell2cell,
                self.neighbor_table_cell2grid,
                self.grid_dim,
                0.0,
                1.0,
            )
            positions = self.transform_cell_to_world(vertices)
            if len(positions) == 0:
                return None

            mesh = o3d.t.geometry.TriangleMesh(
                positions.detach().cpu().numpy(), triangles.detach().cpu().numpy()
            )
            if color_fn is not None:
                colors = color_fn(positions)
                mesh.vertex["colors"] = colors.detach().cpu().numpy()
            if normal_fn is not None:
                normals = normal_fn(positions)
                mesh.vertex["normals"] = normals.detach().cpu().numpy()
            return mesh


class UnBoundedSparseDenseGrid(SparseDenseGrid):
    """Create an unbounded sparse-dense grid.
    This is typically useful when the scene is not known to be bounded before hand.
    In this case, a unit length for dense cell unit is required.

    Example:
    For a sparse-dense grid with grid_dim=3 and with dense_cell_unit=1/8,
    the space will be allocated per initialization.
    The origin point is adaptive to the input.
     ____ ____ ____
    |-3/8|    |-3/8|
    |-3/8|_ __|-1/8|
    |    |-2/8|    |
    |__ _|-2/8|__ _|
    |    |    |-1/8|
    |____|____|-1/8|___ ___ ___ ___ ___ ___
                   |0,0|   |   |0, |   |   |
                   |___|___|___|3/8|___|___|
                   |   |1/8|   |   |1/8|   |
                   |___|1/8|___|___|4/8|___|
                   |2/8|   |2/8|   |   |2/8|
                   |0__|___|2/8|___|___|5/8|
                               |3/8|   |   |
                               |3/8|___|___|
                               |   |4/8|   |
                               |___|___|___|
                               |   |   |5/8|
                               |___|___|5/8|
    """

    def __init__(
        self,
        in_dim: int,
        num_embeddings: int,
        embedding_dim: int,
        grid_dim: int,
        cell_size: float = 0.01,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ):
        super().__init__(in_dim, num_embeddings, embedding_dim, grid_dim, device)

        self.cell_size = cell_size
        self.transform_world_to_cell = lambda x: x / self.cell_size
        self.transform_cell_to_world = lambda x: x * self.cell_size


class BoundedSparseDenseGrid(SparseDenseGrid):
    """Create a bounded sparse-dense grid.
    This is typically useful when the scene is known to be bounded before hand.

    The unit of the grid is the normalized length of a grid cell.

    Example:
    For a sparse-dense grid bounded within [0,0]--[1,1] with grid_dim=3 and sparse_grid_dim=3,
    the voxel unit length is 1 / (3 * 3 - 1) = 1 / 8.
    The origin point will be fixed at (0, 0) of the bounding box.

    (0, 0)                               (0, 1)
     ------------------------------------->
    |0,0|   |   |
    |___|___|___|
    |   |1/8|   |
    |___|1/8|___|
    |   |   |2/8|
    |___|___|2/8|___ ___ ___ ___ ___ ___
    |           |3/8|   |   |3/8|   |   |
    |           |3/8|___|___|6/8|___|___|
    |           |   |4/8|   |   |4/8|   |
    |           |___|4/8|___|___|7/8|___|
    |           |   |   |5/8|   |   |5/8|
    |           |___|___|5/8|___|___|8/8|
    |                       |6/8|   |   |
    |                       |6/8|___|___|
    |                       |   |7/8|   |
    |                       |___|7/8|___|
    |                       |   |   |8/8|
    |                       |___|___|8/8|
    v
    (1, 0)

    """

    def __init__(
        self,
        in_dim: int,
        num_embeddings: int,
        embedding_dim: int,
        grid_dim: int = 8,
        sparse_grid_dim: int = 128,
        bbox_min: torch.Tensor = None,
        bbox_max: torch.Tensor = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        assert in_dim == 3, "Only 3D is supported now"
        super().__init__(in_dim, num_embeddings, embedding_dim, grid_dim, device)

        self.sparse_grid_dim = sparse_grid_dim
        self.bbox_min = (
            bbox_min.to(self.device)
            if bbox_min is not None
            else -1 * torch.ones(3, device=self.device)
        )
        self.bbox_max = (
            bbox_max.to(self.device)
            if bbox_max is not None
            else torch.ones(3, device=self.device)
        )

        self.cell_size = (self.bbox_max - self.bbox_min) / (
            sparse_grid_dim * grid_dim - 1
        )
        self.transform_world_to_cell = lambda x: (x - self.bbox_min) / self.cell_size
        self.transform_cell_to_world = lambda x: x * self.cell_size + self.bbox_min
        # TODO(wei): contraction

    def grids_in_bound(self, grid_coords: torch.Tensor) -> torch.Tensor:
        masks = ((grid_coords >= 0) * (grid_coords < self.sparse_grid_dim)).all(dim=-1)
        return grid_coords[masks]

    def full_init_(self):
        grid_coords = torch.stack(
            torch.meshgrid(*[torch.arange(self.sparse_grid_dim)] * 3)
        ).reshape(3, -1).T.to(self.device)

        self.engine.insert_keys(grid_coords)

