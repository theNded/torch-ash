from typing import List, Union, Tuple, Dict, OrderedDict, Optional, Literal, overload

import torch
import torch.nn as nn
from .core import ASHEngine, ASHModule
from .sparsedense_grid import BoundedSparseDenseGrid, UnBoundedSparseDenseGrid


def _check_type_and_len(arg, length, name: str):
    if isinstance(arg, int):
        return [arg] * length
    elif isinstance(arg, list):
        assert len(arg) == length, f"{name} should be a list of length {length}"
        return arg
    else:
        raise TypeError(f"{name} should be int or list of int")


class UnBoundedMultiResGrid(nn.Module):
    def __init__(
        in_dim: int,
        num_embeddings: List[int] = [512, 2048, 8196, 32768],
        embedding_dims: Union[List[int], int] = 2,
        grid_dims: Union[List[int], int] = 2,
        cell_sizes: List[float] = [0.08, 0.04, 0.02, 0.01],
        device: Optional[Union[str, torch.device]] = torch.device("cpu"),
    ):
        num_levels = len(num_embeddings)
        self.num_embeddings = num_embeddings
        self.embedding_dims = _check_type_and_len(
            embedding_dims, num_levels, "embedding_dims"
        )
        self.grid_dims = _check_type_and_len(grid_dims, num_levels, "grid_dims")
        self.cell_sizes = _check_type_and_len(cell_sizes, num_levels, "cell_sizes")
        for i in range(num_levels):
            grid = UnBoundedSparseDenseGrid(
                in_dim,
                self.num_embeddings[i],
                self.embedding_dims[i],
                self.grid_dims[i],
                self.cell_sizes[i],
                device,
            )
            self.grids.append(grid)


class BoundedMultiResGrid(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_embeddings: List[int] = [512, 2048, 8196, 32768],
        embedding_dims: Union[List[int], int] = 2,
        grid_dims: Union[List[int], int] = 2,
        sparse_grid_dims: Union[List[int], int] = [8, 16, 32, 64],
        bbox_min: torch.Tensor = None,
        bbox_max: torch.Tensor = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Multi-res grid is a simple wrapper of SparseDenseGrid and may have reduced performance.
        It only supports the simplified forward workflow for training neural networks.

        For more fine-grained control, please use grids[level] directly.

        The default setup explained:
        - A 8x8x8 grid, each is a grid of 2x2x2. Assume they occupy the entire space (8^3 = 512)
        - A 16x16x16 grid, each is a grid of 2x2x2. Assume they occupy half the space (16^3 / 2 = 2048)
        - A 32x32x32 grid, each is a grid of 2x2x2. Assume they occupy 1/4 the space (32^3 / 4 = 8192)
        - A 64x64x64 grid, each is a grid of 2x2x2. Assume they occupy 1/8 the space (64^3 / 8 = 32768)
        TODO: optimize necessary OPs in kernels
        """
        assert in_dim == 3, "Only 3D is supported now"
        super().__init__()

        num_levels = len(num_embeddings)

        self.num_embeddings = num_embeddings
        self.embedding_dims = _check_type_and_len(
            embedding_dims, num_levels, "embedding_dims"
        )
        self.grid_dims = _check_type_and_len(grid_dims, num_levels, "grid_dims")
        self.sparse_grid_dims = _check_type_and_len(
            sparse_grid_dims, num_levels, "sparse_grid_dims"
        )

        self.grids = []
        for i in range(num_levels):
            grid = BoundedSparseDenseGrid(
                in_dim,
                self.num_embeddings[i],
                self.embedding_dims[i],
                self.grid_dims[i],
                self.sparse_grid_dims[i],
                bbox_min,
                bbox_max,
                device,
            )
            self.grids.append(grid)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        In the multi-level setup, we do not give masks. The found features are simply concatenated, and
        the empty-space queries directly return zeros.
        """
        features = []
        masks = None
        for grid in self.grids:
            feature, mask = grid(x)
            if masks is None:
                masks = mask
            else:
                masks = masks * mask
            features.append(feature)
        features = torch.cat(features, dim=-1)
        features = torch.where(masks.view(-1, 1), features, torch.zeros_like(features))
        return features, masks

    @torch.no_grad()
    def spatial_init_(self, x: torch.Tensor, dilation: int = 1) -> None:
        for grid in self.grids:
            grid.spatial_init_(x, dilation)

    @torch.no_grad()
    def full_init_(self):
        for grid in self.grids:
            grid.full_init_()
