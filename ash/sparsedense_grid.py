from typing import List, Union, Tuple, Dict, OrderedDict, Optional, Literal, overload

import torch
import torch.nn as nn
from .core import ASHEngine, ASHModule
from .hashmap import HashMap, HashSet

from .common import _get_c_extension

backend = _get_c_extension()


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
    """Query the embeddings given x defined by offsets, sparse_indices, and dense_indices.
    The overall idea is to compute:
    y, z = f(x), f'(x)

    y, z = zeros(), zeros()
    for nb in nbs:
        feature_nb = embeddings[sparse_neighbor_indices_table[sparse_indices, sparse_offset(nb)],
                                dense_neighbor_indices_table[dense_indices, nb]]
        y += weight(x_offsets[nb]) * feature_nb
        z += dweight_dx(x_offsets[nb]) * feature_nb
    return output
    Args:
        embeddings: (num_embeddings, cells_per_dense_grid, embedding_dim) tensor.
        sparse_indices: (num_x) tensor with sparse indices.
        dense_indices: (num_x) tensor with dense indices.
        offsets: (num_x, in_dim) tensor with offsets at the unit of cells, used for interpolation.
        sparse_neighbor_indices_table: (num_embeddings, 8) tensor indices of sparse neighbor grids.
        dense_neighbor_indices_table: (num_embeddings, 8) tensor indices of dense neighbor cells.
    Returns:
        y: (num_x, embedding_dim) tensor. Interpoated embedding.
        z: (num_x, in_dim, embedding_dim) tensor. Closed-form gradient of feature w.r.t. embedding.
    """

    @staticmethod
    def forward(
        ctx,
        embeddings: torch.Tensor,
        offsets: torch.Tensor,
        sparse_indices: torch.Tensor,
        dense_indices: torch.Tensor,
        dense_coords: torch.Tensor,
        masks: torch.Tensor,
        sparse_neighbor_indices_table: torch.Tensor,
        dense_neighbor_indices_table: torch.Tensor,
        dense_grid_dim: int,
    ) -> torch.Tensor:
        ctx.save_for_backward(
            embeddings,
            offsets,
            sparse_indices,
            dense_indices,
            dense_coords,
            masks,
            sparse_neighbor_indices_table,
            dense_neighbor_indices_table,
        )

        y = backend.query_forward(
            embeddings,
            offsets,
            sparse_indices,
            dense_indices,
            dense_coords,
            masks,
            sparse_neighbor_indices_table,
            dense_neighbor_indices_table,
            dense_grid_dim,
        )
        return y

    @staticmethod
    def backward(ctx, z: torch.Tensor):
        (
            embeddings,
            offsets,
            sparse_indices,
            dense_indices,
            sparse_neighbor_indices_table,
            dense_neighbor_indices_table,
            masks,
        ) = ctx.saved_tensors

        dLdembedding, dLdoffsets = SparseDenseGridQueryBackward.apply(
            z,
            embeddings,
            offsets,
            sparse_indices,
            dense_indices,
            masks,
            sparse_neighbor_indices_table,
            dense_neighbor_indices_table,
        )
        return dLdembedding, dLdoffsets, None, None, None, None, None


class SparseDenseGridQueryBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z,
        embeddings,
        offsets,
        sparse_indices,
        dense_indices,
        masks,
        sparse_neighbor_indices_table,
        dense_neighbor_indices_table,
    ):
        ctx.save_for_backward(
            embeddings,
            offsets,
            sparse_indices,
            dense_indices,
            sparse_neighbor_indices_table,
            dense_neighbor_indices_table,
            masks,
        )

        w1, w2 = backend.sparsedense_grid_query_backward_forward(
            z,
            embeddings,
            offsets,
            sparse_indices,
            dense_indices,
            masks,
            sparse_neighbor_indices_table,
            dense_neighbor_indices_table,
        )
        return w1, w2

    @staticmethod
    def backward(ctx, v1: torch.Tensor, v2: torch.Tensor):
        (
            embeddings,
            offsets,
            sparse_indices,
            dense_indices,
            sparse_neighbor_indices_table,
            dense_neighbor_indices_table,
            masks,
        ) = ctx.saved_tensors

        backend.sparse_dense_grid_query_backward_backward(
            v1,
            embeddings,
            offsets,
            sparse_indices,
            dense_indices,
            masks,
            sparse_neighbor_indices_table,
            dense_neighbor_indices_table,
        )
        return dLdembedding, None, None, None, None, None, None, None


class SparseDenseGrid(ASHModule):
    """
    Create an embedding of (num_embeddings, grid_dim^in_dim * dense_grid_dim).
    The embedding is a sparse-dense grid, where each [grid] is a dense [array of cells].

    The unit of input is measured by the number of cells.
    Here is the coordinate convention of a sparse-dense grid with dense_grid_dim = 3.
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
        dense_grid_dim: int,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ):
        super().__init__()

        self.transform_world_to_cell = lambda x: x
        self.transform_cell_to_world = lambda x: x

        self.in_dim = in_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.dense_grid_dim = dense_grid_dim
        self.device = isinstance(device, str) and torch.device(device) or device

        self.engine = ASHEngine(in_dim, num_embeddings, device)
        self.cells_per_dense_grid = dense_grid_dim**in_dim
        self.embeddings = nn.Parameter(
            torch.zeros(
                num_embeddings, self.cells_per_dense_grid, embedding_dim, device=device
            )
        )

        # 8 neighbors for trilinear interpolation reshaped for boradcasting
        self.neighbor_coord_offsets = (
            enumerate_neighbors(in_dim, 1, False).to(self.device).view(1, -1, in_dim)
        )

        # Dense grid look up tables for each cell and their neighbors
        self.dense_indices = torch.arange(
            self.cells_per_dense_grid, dtype=torch.long, device=self.device
        )
        self.dense_coords = self._delinearize_dense_indices(self.dense_indices)
        assert self.dense_coords.shape == (self.cells_per_dense_grid, in_dim)

        dense_neighbor_coords = (
            self.dense_coords.view(-1, 1, in_dim) + self.neighbor_coord_offsets
        ).view(-1, 3) % dense_grid_dim
        self.dense_neighbor_indices_table = self._linearize_dense_coords(
            dense_neighbor_coords
        ).view(self.cells_per_dense_grid, -1)

        assert self.dense_neighbor_indices_table.shape == (
            self.cells_per_dense_grid,
            2**in_dim,
        )

        # Sparse grid look up tables for each grid and their neighbors
        # Need to be constructed after spatial intialization
        self.sparse_coords = None
        self.sparse_neighbor_indices_table = None

    @torch.no_grad()
    def _linearize_dense_coords(self, dense_coords: torch.Tensor) -> torch.Tensor:
        assert len(dense_coords.shape) == 2 and dense_coords.shape[1] == self.in_dim

        """Convert dense coordinates to dense indices."""
        dense_indices = torch.zeros_like(
            dense_coords[:, 0], dtype=torch.long, device=self.device
        )
        for i in range(self.in_dim):
            dense_indices += dense_coords[:, i] * self.dense_grid_dim**i
        return dense_indices.long()

    @torch.no_grad()
    def _delinearize_dense_indices(self, dense_indices: torch.Tensor) -> torch.Tensor:
        """Convert dense indices to dense coordinates."""
        dense_coords = []
        dense_indices_iter = dense_indices.clone()
        for i in range(self.in_dim):
            dense_coords.append(dense_indices_iter % self.dense_grid_dim)
            dense_indices_iter = torch.div(
                dense_indices_iter, self.dense_grid_dim, rounding_mode="floor"
            )
        dense_coords = torch.stack(dense_coords, dim=1)
        return dense_coords.int()

    @torch.no_grad()
    def construct_sparse_neighbor_tables_(self, radius=1, bidirection=False) -> None:
        """Construct a neighbor lookup table for the sparse-dense grid.
        This should only be called once the sparse grid is initialized.
        Used for trilinear interpolation in query and marching cubes.
        Updates:
            self.sparse_coords: (num_embeddings, in_dim)
            self.sparse_neighbor_indices_table: (num_embeddings, 2**in_dim)
        For non-active entries, the neighbor indices are set to -1.
        """
        active_sparse_coords, active_sparse_indices = self.engine.items()

        active_sparse_neighbor_coords = (
            active_sparse_coords.view(-1, 1, self.in_dim).int()
            + self.neighbor_coord_offsets.view(1, -1, self.in_dim).int()
        ).view(-1, self.in_dim)

        # (N*2**in_dim, ), (N*2**in_dim, )
        active_sparse_neighbor_indices, active_sparse_neighbor_masks = self.engine.find(
            active_sparse_neighbor_coords
        )
        # Set not found neighbor indices to -1
        active_sparse_neighbor_indices[~active_sparse_neighbor_masks] = -1

        # Create a dense lookup table for sparse coords and neighbor indices
        self.sparse_coords = torch.empty(
            (self.num_embeddings, self.in_dim), dtype=torch.int32, device=self.device
        )
        self.sparse_coords.fill_(-1)
        self.sparse_coords[active_sparse_indices] = active_sparse_coords

        self.sparse_neighbor_indices_table = torch.empty(
            self.num_embeddings, 2**self.in_dim, dtype=torch.int64, device=self.device
        )
        self.sparse_neighbor_indices_table.fill_(-1)
        self.sparse_neighbor_indices_table[
            active_sparse_indices
        ] = active_sparse_neighbor_indices.view(-1, 8)

    def query(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns:
        sparse_indices: (N, 1) tensor of sparse grid indices
        dense_indices: (N, 1) tensor of dense grid indices
        offsets: (N, in_dim) tensor of offsets in dense grid units
        """
        # TODO: merge in one pass for efficiency if necessary
        x_sparse = (x / self.dense_grid_dim).floor()
        x_dense = x - x_sparse * self.dense_grid_dim
        offsets = x_dense - x_dense.floor()

        sparse_indices, masks = self.engine.find(x_sparse.int())
        dense_indices = self._linearize_dense_coords(x_dense.int())

        return sparse_indices, dense_indices, x_dense.floor().int(), offsets, masks

    def forward(
        self,
        x: torch.Tensor,
        compute_grad_x: bool = False,
        interpolation: Literal["nearest", "linear"] = "linear",
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Used for differentiable query at floating-point locations.
        Args:
            x: (N, in_dim) tensor of keys, converted to the unit of dense grid cells.
        Returns:
            values: (N, embedding_dim) tensor of features
            masks: (N, 1) tensor of masks
        if compute_grad_x, also returns:
            grad_x: (N, embedding_dim, in_dim) tensor of gradients
        """
        x = self.transform_world_to_cell(x)

        sparse_indices, dense_indices, dense_coords, offsets, masks = self.query(x)

        if interpolation == "nearest":
            assert compute_grad_x is False
            return self.embeddings[sparse_indices, dense_indices], masks

        elif interpolation == "linear":
            features = SparseDenseGridQuery.apply(
                self.embeddings,
                offsets,
                sparse_indices,
                dense_indices,
                dense_coords,
                masks,
                self.sparse_neighbor_indices_table,
                self.dense_neighbor_indices_table,
                self.dense_grid_dim
            )
            return features, masks
        else:
            raise ValueError(f"Unknown interpolation: {interpolation}")

    def items(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Used for direct assignment, e.g. initialization with fusion, marching cubes.
        Queries happen at integer locations.

        Here M = dense_grid_dim**in_dim (items per dense grid)
        Returns:
            sparse_coords: (N, 1, in_dim) tensor of keys
            dense_coords: (1, M, in_dim) tensor of keys
            sparse_indices: (N, 1) tensor of features
            dense_indices: (1, M) tensor of features

        For compactness, the coordinates are viewed with the
        (sparse-dim, dense-dim, [embedding-dim]) convention for easier broadcasting.

        The coordinates in the sparse-dense system can be computed by
            x = sparse_coords * dense_grid_dim + dense_coords
        Their associated embedding can be accessed by
            feats = self.embeddings[sparse_indices, dense_indices]
        """
        sparse_coords, sparse_indices = self.engine.items()

        return (
            sparse_coords.view(-1, 1, self.in_dim),
            self.dense_coords.view(1, -1, self.in_dim),
            sparse_indices.view(-1, 1),
            self.dense_indices.view(1, -1),
        )

    def cell_to_world(
        self, sparse_coords: torch.Tensor, dense_coords: torch.Tensor
    ) -> torch.Tensor:
        """Converts cell coordinates to world coordinates.
        Args:
            sparse_coords: (N, 1, in_dim) tensor of sparse coordinates
            dense_coords: (1, M, in_dim) tensor of dense coordinates
        Returns:
            world_coords: (N, M, in_dim) tensor of world coordinates
        """
        return self.transform_cell_to_world(
            sparse_coords.view(-1, 1, self.in_dim) * self.dense_grid_dim
            + dense_coords.view(1, -1, self.in_dim)
        ).view(-1, self.in_dim)

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

        sparse_keys = torch.floor(points / self.dense_grid_dim).int()
        neighbor_coord_offsets = enumerate_neighbors(
            self.in_dim, dilation, bidirectional=False
        ).to(self.device)
        sparse_keys_with_neighbors = (
            sparse_keys.view(-1, 1, 3) + neighbor_coord_offsets
        ).view(-1, 3)

        # No need to use a huge hash set due to the duplicates
        hash_set = HashSet(key_dim=3, capacity=len(sparse_keys), device=self.device)
        hash_set.insert(sparse_keys_with_neighbors)
        unique_sparse_coords = hash_set.keys()

        self.engine.insert_keys(unique_sparse_coords)
        sparse_indices, masks = self.engine.find(unique_sparse_coords)
        assert masks.all()

        return (
            unique_sparse_coords.view(-1, 1, self.in_dim),
            self.dense_coords.view(1, -1, self.in_dim),
            sparse_indices.view(-1, 1),
            self.dense_indices.view(1, -1),
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
        iso_value: float = 0.0,
        weight_thr: float = 1.0,
        vertices_only=False,
    ) -> torch.Tensor:
        """Extract isosurface from the grid.
        Args:
            tsdfs: (num_embeddings, cells_per_dense_grid) tensor of tsdfs
            weights: (num_embeddings, cells_per_dense_grid) tensor of weights
            sparse_coords: (N, 3) tensor of sparse coordinates
            sparse_indices: (N, 1) tensor of sparse indices, N <= num_embeddings
            sparse_neighbor_indices: (N, 2**3) tensor of sparse neighbor indices. -1 means no neighbor
            iso_value: iso value to extract surface
            weight_thr: weight threshold to consider a cell as occupied
        """

        if self.sparse_coords is None or self.sparse_neighbor_indices_table is None:
            self.construct_sparse_neighbor_tables_()

        # Get active entries
        sparse_coords, sparse_indices = self.engine.items()

        if vertices_only:
            vertices = backend.isosurface_extraction(
                tsdfs,
                weights,
                sparse_indices,
                self.sparse_coords,
                self.sparse_neighbor_indices_table,
                self.dense_coords,
                self.dense_neighbor_indices_table,
                self.dense_grid_dim,
                0.0,
                1.0,
            )
            return self.transform_cell_to_world(vertices)

        else:
            triangles, vertices = backend.marching_cubes(
                tsdfs,
                weights,
                sparse_indices,
                self.sparse_coords,
                self.sparse_neighbor_indices_table,
                self.dense_coords,
                self.dense_neighbor_indices_table,
                self.dense_grid_dim,
                0.0,
                1.0,
            )
            return triangles, self.transform_cell_to_world(vertices)


class BoundedSparseDenseGrid(SparseDenseGrid):
    """Create a bounded sparse-dense grid.
    This is typically useful when the scene is known to be bounded before hand.

    The unit of the grid is the normalized length of a grid cell.

    Example:
    For a sparse-dense grid bounded within [0,0]--[1,1] with dense_grid_dim=3 and sparse_grid_dim=3,
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
        dense_grid_dim: int = 8,
        sparse_grid_dim: int = 512,
        bbox_min: torch.Tensor = torch.zeros(3),
        bbox_max: torch.Tensor = torch.ones(3),
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__(in_dim, num_embeddings, embedding_dim, dense_grid_dim, device)

        self.cell_size = (bbox_max - bbox_min) / (sparse_grid_dim * dense_grid_dim - 1)
        self.transform_world_to_cell = lambda x: (x - bbox_min) / self.cell_size
        self.transform_cell_to_world = lambda x: x * self.cell_size + bbox_min
        # TODO(wei): contraction


class UnBoundedSparseDenseGrid(SparseDenseGrid):
    """Create an unbounded sparse-dense grid.
    This is typically useful when the scene is not known to be bounded before hand.
    In this case, a unit length for dense cell unit is required.

    Example:
    For a sparse-dense grid with dense_grid_dim=3 and with dense_cell_unit=1/8,
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
        dense_grid_dim: int,
        dense_cell_size: float = 0.01,
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ):
        super().__init__(in_dim, num_embeddings, embedding_dim, dense_grid_dim, device)

        self.cell_size = dense_cell_size
        self.transform_world_to_cell = lambda x: x / self.cell_size
        self.transform_cell_to_world = lambda x: x * self.cell_size
