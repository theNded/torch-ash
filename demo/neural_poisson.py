from typing import Tuple

import argparse

import torch
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

from ash import UnBoundedSparseDenseGrid, BoundedSparseDenseGrid, DotDict
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

import open3d as o3d
import open3d.core as o3c


class PointCloudDataset(torch.utils.data.Dataset):
    """Minimal point cloud dataset for a single point cloud"""

    def __init__(self, path, normalize_scene=True):
        self.path = Path(path)

        self.pcd = o3d.t.io.read_point_cloud(str(self.path))

        self.positions = self.pcd.point.positions.numpy().astype(np.float32)
        assert "normals" in self.pcd.point
        self.normals = self.pcd.point.normals.numpy().astype(np.float32)

        assert len(self.positions) == len(self.normals)

        min_vertices = np.min(self.positions, axis=0)
        max_vertices = np.max(self.positions, axis=0)

        self.center = (min_vertices + max_vertices) / 2.0
        self.scale = 2.0 / (np.max(max_vertices - min_vertices) * 1.1)

        self.positions = (self.positions - self.center) * self.scale

        # For visualization
        self.pcd = o3d.t.geometry.PointCloud(self.positions)
        self.pcd.point.normals = self.normals

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return {
            "position": self.positions[idx],
            "normal": self.normals[idx],
        }


class NeuralPoisson(torch.nn.Module):
    def __init__(self, grid_resolution, embedding_dim, device=torch.device("cuda:0")):
        super().__init__()

        # Directly map from position to density
        self.grid = BoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=10000,
            embedding_dim=embedding_dim,
            grid_dim=16,
            sparse_grid_dim=grid_resolution,
            device=device,
        )
        self.bbox_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
        )

    @torch.no_grad()
    def spatial_init_(self, positions: torch.Tensor):
        self.grid.spatial_init_(positions, dilation=1)

    def sample_grid(self, num_samples=10000):
        grid_coords, cell_coords, grid_indices, cell_indices = self.grid.items()
        grid_sel = torch.randint(
            0, len(grid_coords), (num_samples, 1), device=grid_coords.device
        )

        rand_grid_coords = grid_coords[grid_sel].view(-1, 3)
        rand_offsets = torch.rand(num_samples, 3, device=grid_coords.device) * (
            self.grid.cell_size * self.grid.grid_dim
        )

        rand_positions = self.grid.transform_cell_to_world(
            rand_grid_coords * self.grid.grid_dim + rand_offsets
        )
        return rand_positions

    @torch.no_grad()
    def visualize_occupied_cells(self):
        grid_coords, cell_coords, grid_indices, cell_indices = self.grid.items()
        cell_positions = self.grid.cell_to_world(grid_coords, cell_coords)
        pcd = o3d.t.geometry.PointCloud(cell_positions.cpu().numpy())
        pcd.paint_uniform_color([0.0, 0.0, 1.0])
        return pcd


class NeuralPoissonPlain(NeuralPoisson):
    def __init__(self, grid_resolution, device=torch.device("cuda:0")):
        super().__init__(
            grid_resolution=grid_resolution, embedding_dim=1, device=device
        )
        torch.nn.init.zeros_(self.grid.embeddings)

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions.requires_grad_(True)
        embedding, mask = self.grid(positions)

        grad_x = torch.autograd.grad(
            outputs=embedding[..., 0],
            inputs=positions,
            grad_outputs=torch.ones_like(embedding[..., 0], requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return embedding, grad_x, mask

    def marching_cubes(self):
        def normal_fn(positions):
            positions.requires_grad_(True)
            density, grad_x, mask = self.forward(positions)
            return grad_x

        mesh = self.grid.marching_cubes(
            tsdfs=self.grid.embeddings[..., 0].contiguous(),
            weights=2 * torch.ones_like(self.grid.embeddings[..., 0]).contiguous(),
            color_fn=None,
            normal_fn=normal_fn,
            iso_value=0.0,
            vertices_only=False,
        )
        return mesh


class NeuralPoissonMLP(NeuralPoisson):
    def __init__(self, grid_resolution, device=torch.device("cuda:0")):
        super().__init__(
            grid_resolution=grid_resolution, embedding_dim=8, device=device
        )

        torch.nn.init.uniform_(self.grid.embeddings)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1),
        ).to(device)

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions.requires_grad_(True)
        embedding, mask = self.grid(positions)
        sdf = self.mlp(embedding)

        grad_x = torch.autograd.grad(
            outputs=sdf[...,0],
            inputs=positions,
            grad_outputs=torch.ones_like(sdf[...,0], requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return sdf, grad_x, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    dataset = PointCloudDataset(args.path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=True)

    model = NeuralPoissonMLP(grid_resolution=64)
    model.spatial_init_(torch.from_numpy(dataset.positions).cuda())
    print(model.grid.engine.size())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # mesh = model.marching_cubes()
    # if mesh is not None:
    #     o3d.io.write_triangle_mesh(f"mesh_init.ply", mesh.to_legacy())

    # o3d.visualization.draw(
    #     [model.bbox_lineset, model.visualize_occupied_cells(), dataset.pcd]
    # )

    for epoch in range(20):
        pbar = tqdm(dataloader)

        for batch in pbar:
            optimizer.zero_grad()

            positions = batch["position"].cuda()
            normals = batch["normal"].cuda()

            sdf, grad_x, mask = model(positions)

            loss_surface = 100 * sdf[mask].pow(2).mean()
            loss_normal = (grad_x - normals)[mask].pow(2).sum(dim=-1).mean()

            rand_positions = model.sample_grid(num_samples=2 * len(positions))
            sdf_rand, grad_x_rand, mask_rand = model(rand_positions)
            loss_eikonal = (
                (torch.norm(grad_x_rand[mask_rand], dim=-1) - 1).pow(2).mean()
            )

            loss = loss_surface + loss_normal + loss_eikonal
            loss.backward()

            pbar.set_description(
                f"Total={loss.item():.4f}, surface={loss_surface.item():.4f}, normal={loss_normal.item():.4f}, eikonal={loss_eikonal.item():.4f}"
            )
            optimizer.step()

        # mesh = model.marching_cubes()
        # if mesh is not None:
        #     o3d.io.write_triangle_mesh(f"mesh_{epoch:03d}.ply", mesh.to_legacy())
