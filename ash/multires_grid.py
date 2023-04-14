from typing import List, Union, Tuple, Dict, OrderedDict, Optional, Literal, overload

import torch
import torch.nn as nn
from .core import ASHEngine, ASHModule


class MultiResGrid(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_embeddings: int,
        embedding_dims: int,
        min_grid_resolution: int = 16,  # inclusive
        max_grid_resolution: int = 2048,  # inclusive
        levels: int = 9,
    ):
        """
        Create an embedding of (num_embeddings, embedding_dim * levels)

        hash_fn: leveled

        level_factor = 2**(log2(max_grid_resolution / min_grid_resolution) / (levels - 1))
        The size of the multires grids are (1 / min_grid_resolution, 1 / (min_grid_resolution * level_factor), ..., 1 / max_grid_resolution.)

        Performance is not guaranteed for this module. Use tinycudacnn if slow/unstable.
        """
        super().__init__()

        self.in_dim = in_dim

        # Additional dimension for level
        self.engine = ASHEngine(1 + in_dim, num_embeddings)
        self.embeddings = nn.Parameter(
            torch.zeros(num_embeddings + 1, levels * embedding_dim)
        )

    def forward(
        self, keys: torch.Tensor, interp: Literal["nearest", "linear"] = "linear"
    ) -> torch.Tensor:
        """
        Interpolation: leveled sparse neighbors
        neighbor search in hash map: (l, x, y, z) -> (l, x + dx, y + dy, z + dz)
        """
        pass

    # Geometry-based initialization
    def ray_init_(
        self,
        ray_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_near: torch.Tensor,
        rays_far: torch.Tensor,
        dilation: int = 3,
    ) -> None:
        pass

    def spatial_init_(
        self,
        keys: torch.Tensor,
        dilations: Union[List[int], int] = 3,  # corresponds to 3x3x3
    ) -> None:
        """
        Dilate the voxels by the given dilations at each level, and insert them into the hash map respectively.
        If a list is given, len(dilations) == len(grid_dims)
        """
        pass

    # Sampling
    def ray_sample(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_near: torch.Tensor,
        rays_far: torch.Tensor,
        num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Following nerfacc to reuse its rendering
        pass

    def uniform_sample(
        self, num_samples: int, space: Literal["occupied", "empty"] = "occupied"
    ) -> torch.Tensor:
        pass
