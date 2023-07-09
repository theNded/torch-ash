from typing import (
    Union,
    Tuple,
    Optional,
    Literal,
)

import torch
import torch.nn as nn

from .grid_query import SparseDenseGridQuery
from .grid_nns import enumerate_neighbor_coord_offsets

from .core import ASHEngine, ASHModule
from .hashmap import HashSet

from .common import _get_c_extension

backend = _get_c_extension()

"""
General dtype rules:
- coords in world: float
- coords in cells: int
- indices: long
"""

class SparseDenseGrid(ASHModule):
    """
    Create a sparse-dense grid, where each [grid] is a dense array of [cells].

    In the data perspective, it is stored as a tensor of shape (num_grids, cells_per_grid, embedding_dim),
    where cells_per_grid = grid_dim**3.

    In the spatial perspective, it is a grid of cells. Each cell can be queried by
    - grid_idx, cell_idx = grid.query(position)
    - output = grid.embeddings[grid_idx, cell_idx]
    Where the unit of input 'position' is measured by the length of a cell.

    Here is an example of a sparse-dense grid with grid_dim = 3.
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

        # Cache useful constants to avoid duplicate computation in kernels

        # Dense cells have a fixed structure and their properties
        # can be constructed at initialization.
        self.cell_indices = torch.arange(
            self.num_cells_per_grid, dtype=torch.long, device=self.device
        )

        self.cell_coords = self._delinearize_cell_indices(self.cell_indices)
        assert self.cell_coords.shape == (self.num_cells_per_grid, in_dim)

        (
            self.lut_cell_nb2grid_nb,
            self.lut_cell_nb2cell_idx,
        ) = self.construct_cell_neighbor_lut(radius=1, bidirectional=False)

        # Sparse grids are active entries and can only be acquired at run time.
        self.grid_indices = None
        self.grid_coords = None
        self.lut_grid_nb2grid_idx = None

    ###
    # Sparse grid allocation
    ###
    def spatial_init_(
        self, points: torch.Tensor, dilation: int = 1, bidirectional: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        grid_nb_coord_offsets, _ = enumerate_neighbor_coord_offsets(
            self.in_dim, dilation, bidirectional=bidirectional, device=self.device
        )
        grid_nb_coords = (grid_coords.view(-1, 1, 3) + grid_nb_coord_offsets).view(
            -1, 3
        )

        # No need to use a huge hash set due to the duplicates
        hash_set = HashSet(key_dim=3, capacity=len(grid_coords), device=self.device)
        hash_set.insert(grid_nb_coords)
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
        raise NotImplementedError("Ray-tracing initialization not implemented yet")

    ###
    # Core sparse-dense query and enumeration
    ###
    def query(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        grid_indices, cell_indices, offsets, masks = self.query(x)

        if interpolation == "nearest":
            return self.embeddings[grid_indices, cell_indices], masks

        elif interpolation in ["linear", "smooth_step"]:
            if self.grid_coords is None or self.lut_grid_nb2grid_idx is None:
                self.construct_grid_neighbor_lut_(radius=1, bidirectional=False)

            features = SparseDenseGridQuery.apply(
                self.embeddings,
                offsets,
                grid_indices,
                cell_indices,
                masks,
                self.lut_grid_nb2grid_idx,
                self.lut_cell_nb2cell_idx,
                self.lut_cell_nb2grid_nb,
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

    ###
    # Geometric and index transform
    ###
    @torch.no_grad()
    def get_bbox(self):
        return self.bbox_min, self.bbox_max

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

    ###
    # Sampling
    ###
    @torch.no_grad()
    def ray_sample(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        t_min: float,
        t_max: float,
        t_step: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the sparse-dense grid along rays.
        Args:
            rays_o: (N, in_dim) tensor of ray origins
            rays_d: (N, in_dim) tensor of ray directions
            bbox_min: (in_dim,) min clipping box in 3d
            bbox_max: (in_dim,) max clipping box in 3d
            t_min: near clipping distance along ray
            t_max: far clipping distance along ray
            t_step: step size for sampling
        Returns: (similar to nerfacc)
            ray_indices: (M, 1) associated ray index per sample
            t_near: (M, 1) near range of the interval per sample
            t_far: (M, 1) far range of the interval per sample
            ray_prefix_sum: (N + 1) prefix sum of the number of samples per ray
        Samples can be obtained by
            t = (t_near + t_far) / 2
            x = rays_o[ray_indices] + t * rays_d[ray_indices]
        Rendering could be done by a renderer that interprets ray_prefix_sum.

        Pseudo code:
        rgb, density = rgb_fn(x), density_fn(x)
        for i in range(N):
            result_ray[i] = accumulate(rgb[ray_prefix_sum[i]:ray_prefix_sum[i + 1]],
                                       density[ray_prefix_sum[i]:ray_prefix_sum[i + 1]])

        Note in the backend, everything is using the unit of 1 cell size.
        """
        bbox_min, bbox_max = self.get_bbox()

        ray_indices, t_nears, t_fars, ray_prefix_sum = backend.ray_sample(
            self.engine.backend,
            self.transform_world_to_cell(rays_o),
            rays_d,
            self.transform_world_to_cell(bbox_min),
            self.transform_world_to_cell(bbox_max),
            t_min / self.cell_size,
            t_max / self.cell_size,
            t_step / self.cell_size,
            float(self.grid_dim),
        )

        return (
            ray_indices,
            t_nears * self.cell_size,
            t_fars * self.cell_size,
            ray_prefix_sum,
        )

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
        raise NotImplementedError("uniform_sample not implemented")

    ###
    # Convolution/filter for sparse-dense structure
    ###
    @torch.no_grad()
    def gaussian_filter_(self, size: int, sigma: float) -> None:
        """Gaussian filter.
        Args:
            radius: radius of the filter
            sigma: sigma of the gaussian
        Returns:
            weights: (2R+1, 2R+1, 2R+1) tensor of weights
        """
        assert size % 2 == 1, "size must be odd"

        import scipy.stats as st
        import numpy as np

        def gkern(size, nsig, dim=2):
            x = np.linspace(-nsig, nsig, size + 1)

            kern1d = np.diff(st.norm.cdf(x))
            if dim == 1:
                return torch.from_numpy(kern1d / kern1d.sum()).float()

            kern2d = np.outer(kern1d, kern1d)
            if dim == 2:
                return torch.from_numpy(kern2d / kern2d.sum()).float()

            kern3d = np.outer(kern1d, kern2d).reshape((size, size, size))
            if dim == 3:
                return torch.from_numpy(kern3d / kern3d.sum()).float()

            raise NotImplementedError("Unsupported dimension")

        filter_dim = self.embeddings.shape[-1]
        weights = (
            gkern(size, sigma, dim=3)
            .view(size, size, size, 1)
            .repeat(1, 1, 1, filter_dim)
        )
        weights = weights.to(self.embeddings.device)

        self.embeddings.copy_(self.convolution(self.embeddings, weights))

    def convolution(self, inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # inputs: (num_embeddings, num_cells_per_grid, in_dim)
        # Input shape check -- need to be the same as the grid structure
        assert inputs.is_contiguous()
        assert inputs.shape[:2] == self.embeddings.shape[:2]

        # weights: (2R+1, 2R+1, 2R+1, in_dim, output_dim)
        # Weight shape check
        assert weights.is_contiguous()
        assert len(weights.shape) == 4
        assert weights.shape[0] % 2 == 1
        radius = weights.shape[0] // 2
        for d in range(3):
            assert weights.shape[d] == 2 * radius + 1
        assert weights.shape[3] == inputs.shape[-1]
        assert weights.device == inputs.device

        # The window is (usually) larger than the interpolation window
        (
            conv_lut_cell_nb2grid_nb,
            conv_lut_cell_nb2cell_idx,
        ) = self.construct_cell_neighbor_lut(radius, bidirectional=True)

        grid_radius = (radius + self.grid_dim - 1) // self.grid_dim
        conv_grid_coords, conv_lut_grid_nb2grid_idx = self.construct_grid_neighbor_lut(
            grid_radius, bidirectional=True
        )

        grid_coords, cell_coords, grid_indices, cell_indices = self.items()

        num_cell_nbs = conv_lut_cell_nb2grid_nb.shape[1]
        weights = weights.view(num_cell_nbs, weights.shape[-1])
        output = backend.convolution_forward(
            inputs,
            weights,
            torch.ones_like(inputs, dtype=bool),
            grid_indices,
            cell_indices,
            conv_lut_grid_nb2grid_idx,
            conv_lut_cell_nb2cell_idx,
            conv_lut_cell_nb2grid_nb,
            self.grid_dim,
        )

        return output

    ###
    # Radius neighbor lookup table construction
    ###
    @torch.no_grad()
    def construct_cell_neighbor_lut(
        self, radius: int, bidirectional: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            cell_coord_offsets,
            cell_coord_fn_offset2idx,
        ) = enumerate_neighbor_coord_offsets(
            self.in_dim, radius, bidirectional, self.device
        )

        (
            grid_coord_offsets,
            grid_coord_fn_offset2idx,
        ) = enumerate_neighbor_coord_offsets(
            self.in_dim,
            (radius + self.grid_dim - 1) // self.grid_dim,
            bidirectional,
            self.device,
        )

        assert self.cell_coords is not None
        cell_nb_coords = self.cell_coords.view(
            -1, 1, self.in_dim
        ) + cell_coord_offsets.view(1, -1, self.in_dim)

        cell_nb_coords = cell_nb_coords.view(-1, self.in_dim)

        grid_nb_offsets = torch.div(
            cell_nb_coords, self.grid_dim, rounding_mode="floor"
        ).long()
        cell_nb_coords = torch.remainder(cell_nb_coords, self.grid_dim)
        cell_nb_indices = self._linearize_cell_coords(cell_nb_coords)

        grid_nbs = grid_coord_fn_offset2idx(grid_nb_offsets)

        lut_cell_nb2grid_nb = grid_nbs.view(self.num_cells_per_grid, -1)
        lut_cell_nb2cell_idx = cell_nb_indices.view(self.num_cells_per_grid, -1)

        return lut_cell_nb2grid_nb, lut_cell_nb2cell_idx

    @torch.no_grad()
    def construct_grid_neighbor_lut(
        self, radius: int, bidirectional: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Construct a neighbor lookup table for the sparse-dense grid.
        This should only be called once the sparse grid is initialized.
        Used for trilinear interpolation in query and marching cubes.
        Updates:
            self.grid_coords: (num_embeddings, in_dim)
            self.lut_grid_nb2grid_idx: (num_embeddings, 2**in_dim)
        For non-active entries, the neighbor indices are set to -1.
        """
        active_grid_coords, active_grid_indices = self.engine.items()

        (
            grid_coord_offsets,
            grid_coord_fn_offset2idx,
        ) = enumerate_neighbor_coord_offsets(
            self.in_dim,
            radius,
            bidirectional,
            self.device,
        )
        num_nbs = len(grid_coord_offsets)

        active_grid_nb_coords = (
            active_grid_coords.view(-1, 1, self.in_dim).int()
            + grid_coord_offsets.view(1, -1, self.in_dim).int()
        ).view(-1, self.in_dim)

        # (N*2**in_dim, ), (N*2**in_dim, )
        active_grid_nb_indices, active_grid_nb_masks = self.engine.find(
            active_grid_nb_coords
        )
        # Set not found neighbor indices to -1
        active_grid_nb_indices[~active_grid_nb_masks] = -1

        # Create a dense lookup table for sparse coords and neighbor indices
        grid_coords = torch.empty(
            (self.num_embeddings, self.in_dim), dtype=torch.int32, device=self.device
        )
        grid_coords.fill_(-1)
        grid_coords[active_grid_indices] = active_grid_coords

        lut_grid_nb2grid_idx = torch.empty(
            self.num_embeddings, num_nbs, dtype=torch.int64, device=self.device
        )
        lut_grid_nb2grid_idx.fill_(-1)
        lut_grid_nb2grid_idx[active_grid_indices] = active_grid_nb_indices.view(
            -1, num_nbs
        )
        return grid_coords, lut_grid_nb2grid_idx

    @torch.no_grad()
    def construct_grid_neighbor_lut_(self, radius: int, bidirectional: bool) -> None:
        self.grid_coords, self.lut_grid_nb2grid_idx = self.construct_grid_neighbor_lut(
            radius, bidirectional
        )

    ###
    # Surface extraction
    ###
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

        if self.grid_coords is None or self.lut_grid_nb2grid_idx is None:
            self.construct_grid_neighbor_lut_(radius=1, bidirectional=False)

        # Get active entries
        grid_coords, grid_indices = self.engine.items()

        if vertices_only:
            vertices = backend.isosurface_extraction(
                tsdfs,
                weights,
                grid_indices,
                self.grid_coords,
                self.lut_grid_nb2grid_idx,
                self.cell_coords,
                self.lut_cell_nb2cell_idx,
                self.lut_cell_nb2grid_nb,
                self.grid_dim,
                iso_value,
                weight_thr,
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
                self.lut_grid_nb2grid_idx,
                self.cell_coords,
                self.lut_cell_nb2cell_idx,
                self.lut_cell_nb2grid_nb,
                self.grid_dim,
                iso_value,
                weight_thr,
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

        self.bbox_min = None
        self.bbox_max = None

    @torch.no_grad()
    def compute_bbox_(self):
        active_grid_coords, _ = self.engine.items()

        self.bbox_min = active_grid_coords.min(dim=0, keepdim=True)[0]
        self.bbox_max = active_grid_coords.max(dim=0, keepdim=True)[0] + 1

        self.bbox_min = self.transform_cell_to_world(self.bbox_min[0] * self.grid_dim)
        self.bbox_max = self.transform_cell_to_world(self.bbox_max[0] * self.grid_dim)

    @torch.no_grad()
    def get_bbox(self, force_recompute: bool = False):
        if self.bbox_min is None or self.bbox_max is None or force_recompute:
            self.compute_bbox_()
        return self.bbox_min, self.bbox_max


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
            else -1 * torch.ones(self.in_dim, device=self.device)
        )
        self.bbox_max = (
            bbox_max.to(self.device)
            if bbox_max is not None
            else torch.ones(self.in_dim, device=self.device)
        )

        self.cell_size = (
            ((self.bbox_max - self.bbox_min) / (sparse_grid_dim * grid_dim - 1))
            .min()
            .item()
        )
        self.transform_world_to_cell = lambda x: (x - self.bbox_min) / self.cell_size
        self.transform_cell_to_world = lambda x: x * self.cell_size + self.bbox_min
        # TODO(wei): contraction

    def grids_in_bound(self, grid_coords: torch.Tensor) -> torch.Tensor:
        masks = ((grid_coords >= 0) * (grid_coords < self.sparse_grid_dim)).all(dim=-1)
        return grid_coords[masks]

    def dense_init_(self):
        grid_coords = (
            torch.stack(
                torch.meshgrid(*[torch.arange(self.sparse_grid_dim)] * self.in_dim)
            )
            .reshape(self.in_dim, -1)
            .T.to(self.device)
        )

        self.engine.insert_keys(grid_coords)
