from typing import Union

import torch
import torch.nn as nn

from .core import ASHEngine, ASHModule


class HashEmbedding(ASHModule):
    """
    Aliasing for HashMap, but use different arg names for the constructor.
    Also disable resizing to maintain a static embedding.
    """

    def __init__(
        self,
        in_dim: int,
        num_embeddings: int,
        embedding_dim: int,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        self.embedding = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim, device=device)
        )
        self.engine = ASHEngine(in_dim, num_embeddings, device)

    def spatial_init_(self, keys: torch.Tensor) -> None:
        self.engine.insert_keys(keys)

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        indices, masks = self.engine.find(keys)

        # return 0 for not-found keys
        return self.embedding[indices] * masks.unsqueeze(-1)
