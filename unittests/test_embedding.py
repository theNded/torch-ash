import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from ash import HashEmbedding
import pytest


class TestEmbedding:
    capacity = 1000
    device = torch.device("cuda:0")

    def _generate_keys(self, dim, num):
        keys = (
            torch.arange(num, dtype=torch.int32, device=self.device)
            .view(-1, 1)
            .tile((1, dim))
        )
        perm = torch.randperm(len(keys), device=self.device)
        keys = keys[perm]
        return keys

    def _init_block(self, key_dim, feat_dim):
        embedding = HashEmbedding(key_dim, self.capacity, feat_dim, self.device)

    def _spatial_init_block(self, key_dim, feat_dim):
        embedding = HashEmbedding(key_dim, self.capacity, feat_dim, self.device)

        keys = self._generate_keys(key_dim, self.capacity + 1)
        embedding.spatial_init_(keys[: self.capacity])

    def _forward_block(self, key_dim, feat_dim):
        embedding = HashEmbedding(key_dim, self.capacity, feat_dim, self.device)

        keys = self._generate_keys(key_dim, self.capacity * 2)
        embedding.spatial_init_(keys[: self.capacity])

        output = embedding(keys)
        assert output.shape == (self.capacity * 2, feat_dim)
        assert torch.allclose(
            output[self.capacity :], torch.zeros((1, 1), device=self.device)
        )

    def _backward_block(self, key_dim, feat_dim, tol):
        embedding = HashEmbedding(key_dim, self.capacity, feat_dim, self.device)

        # test correctness with missing keys that should not pollute queries
        num = self.capacity * 2
        keys = self._generate_keys(key_dim, num)

        valid_keys = keys[: self.capacity]
        embedding.spatial_init_(valid_keys)

        def _compute_gt(keys):
            return (keys[:, 0] / self.capacity).view(-1, 1)

        optim = torch.optim.Adam(embedding.parameters(), lr=1e-2)
        batch_size = self.capacity

        with tqdm(range(4000)) as pbar:
            for step in pbar:
                optim.zero_grad()

                sel = torch.randint(0, num, (batch_size,), device=self.device)

                # Heavily polluted input so loss may not go down
                keys_sel = keys[sel]

                output = embedding(keys_sel)
                gt = _compute_gt(keys_sel)

                loss = torch.mean((output - gt) ** 2)
                pbar.set_description(f"loss: {loss.item():.4f}")
                loss.backward()
                optim.step()

        output = embedding(valid_keys)
        gt = _compute_gt(valid_keys)
        diff = (output - gt).abs()
        assert diff.mean().item() < tol
        assert diff.max().item() < tol * 10

        # Reproducible?
        state_dict = embedding.state_dict()
        torch.save(state_dict, "test_embedding.pt")
        del embedding
        del state_dict

        embedding = HashEmbedding(key_dim, self.capacity, feat_dim, self.device)
        state_dict = torch.load("test_embedding.pt", map_location=self.device)
        embedding.load_state_dict(state_dict)

        output = embedding(valid_keys)
        gt = _compute_gt(valid_keys)
        diff = (output - gt).abs()
        assert diff.mean().item() < tol
        assert diff.max().item() < tol * 10

    def test_init(self):
        self._init_block(1, 16)
        self._init_block(2, 16)
        self._init_block(3, 16)
        self._init_block(1, 128)
        self._init_block(2, 128)
        self._init_block(3, 128)

    def test_spatial_init(self):
        self._spatial_init_block(1, 16)
        self._spatial_init_block(2, 16)
        self._spatial_init_block(3, 16)
        self._spatial_init_block(1, 128)
        self._spatial_init_block(2, 128)
        self._spatial_init_block(3, 128)

    def test_forward(self):
        self._forward_block(1, 16)
        self._forward_block(2, 16)
        self._forward_block(3, 16)
        self._forward_block(1, 128)
        self._forward_block(2, 128)
        self._forward_block(3, 128)

    def test_backward(self):
        self._backward_block(1, 16, tol=1e-2)
        self._backward_block(3, 128, tol=1e-2)


TestEmbedding().test_backward()
