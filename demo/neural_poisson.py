from typing import Tuple

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

from ash import UnBoundedSparseDenseGrid, BoundedSparseDenseGrid, DotDict
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

import open3d as o3d
import open3d.core as o3c

from siren import SirenMLP


class PointCloudDataset(torch.utils.data.Dataset):
    """Minimal point cloud dataset for a single point cloud"""

    def __init__(self, path, normalize_scene=True):
        self.path = Path(path)

        self.pcd = o3d.t.io.read_point_cloud(str(self.path))

        self.positions = self.pcd.point.positions.numpy().astype(np.float32)
        assert "normals" in self.pcd.point
        self.normals = self.pcd.point.normals.numpy().astype(np.float32)
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True)

        assert len(self.positions) == len(self.normals)

        min_vertices = np.min(self.positions, axis=0)
        max_vertices = np.max(self.positions, axis=0)

        self.center = (min_vertices + max_vertices) / 2.0
        self.scale = 2.0 / (np.max(max_vertices - min_vertices) * 1.1)

        # Normalize the point cloud into [-1, 1] box
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


class NeuralPoisson(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        grid_dim,
        sparse_grid_dim,
        initialization="random",
        device=torch.device("cuda:0"),
    ):
        super().__init__()

        self.grid = BoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            sparse_grid_dim=sparse_grid_dim,
            device=device,
        )
        if initialization == "random":
            nn.init.normal_(self.grid.embeddings, mean=0.0, std=0.05)
        elif initialization == "zeros":
            nn.init.zeros_(self.grid.embeddings)
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

        # For visualization
        self.bbox_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
        )

    @torch.no_grad()
    def spatial_init_(self, positions: torch.Tensor):
        self.grid.spatial_init_(positions, dilation=1)

    def sample(self, num_samples=10000):
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

    def marching_cubes(self, fname):
        def normal_fn(positions):
            positions.requires_grad_(True)
            sdf, grad_x, mask = self.forward(positions)
            return grad_x

        def sdf_fn():
            grid_coords, cell_coords, grid_indices, cell_indices = self.grid.items()
            cell_positions = self.grid.cell_to_world(grid_coords, cell_coords)
            cell_positions = cell_positions.view(grid_coords.shape[0], -1, 3)

            num_embeddings, cells_per_grid, _ = self.grid.embeddings.shape
            sdfs = torch.zeros(
                num_embeddings,
                cells_per_grid,
                1,
                device=self.grid.embeddings.device,
            )
            weights = torch.zeros_like(sdfs)

            for i in tqdm(range(len(grid_coords))):
                positions = cell_positions[i]
                lhs = grid_indices[i]

                sdf, grad_x, mask = self.forward(positions)
                sdfs[lhs] = sdf.view(-1, cells_per_grid, 1).detach()
                weights[lhs] = 2 * (mask.view(-1, cells_per_grid, 1).detach().float())

            return sdfs, weights

        tsdfs, weights = sdf_fn()
        mesh = self.grid.marching_cubes(
            tsdfs=tsdfs,
            weights=weights,
            color_fn=None,
            normal_fn=normal_fn,
            iso_value=0.0,
            vertices_only=False,
        )
        if mesh is not None:
            o3d.io.write_triangle_mesh(fname, mesh.to_legacy())
        else:
            print("No mesh found.")
        return mesh


class NeuralPoissonPlain(NeuralPoisson):
    def __init__(
        self,
        num_embeddings,
        grid_dim,
        sparse_grid_dim,
        initialization="zeros",
        device=torch.device("cuda:0"),
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=1,
            grid_dim=grid_dim,
            sparse_grid_dim=sparse_grid_dim,
            initialization=initialization,
            device=device,
        )

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions.requires_grad_(True)
        sdf, mask = self.grid(positions)

        grad_x = torch.autograd.grad(
            outputs=sdf[..., 0],
            inputs=positions,
            grad_outputs=torch.ones_like(sdf[..., 0], requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return sdf, grad_x, mask


class NeuralPoissonMLP(NeuralPoisson):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        grid_dim,
        sparse_grid_dim,
        initialization="random",
        num_layers=2,
        hidden_dim=128,
        device=torch.device("cuda:0"),
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            grid_dim=grid_dim,
            sparse_grid_dim=sparse_grid_dim,
            initialization=initialization,
            device=device,
        )

        layers = [nn.Linear(embedding_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        # TODO: geometric initialization? necessary?
        # TODO: skip layers? positional encoding? sin activation?

        self.mlp = nn.Sequential(*layers).to(device)

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions.requires_grad_(True)
        embedding, mask = self.grid(positions)
        sdf = self.mlp(embedding)

        grad_x = torch.autograd.grad(
            outputs=sdf[..., 0],
            inputs=positions,
            grad_outputs=torch.ones_like(sdf[..., 0], requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return sdf, grad_x, mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--model", type=str, default="plain", choices=["plain", "mlp", "siren"])
    parser.add_argument(
        "--grid_dim",
        type=int,
        default=8,
        help="Locally dense grid dimension, e.g. 8 for 8x8x8",
    )
    parser.add_argument(
        "--sparse_grid_dim",
        type=int,
        default=32,
        help="Sparse grid dimension to split the [-1, 1] box. sparse_grid_dim=32 and grid_dim=8 results in a equivalent 256^3 grid, with embeddings only activated at a subset of 32x32x32 sparse grids",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=12000,
        help="expected active sparse voxels in the grid. Worst case: sparse_grid_dim^3",
    )
    # Only used for model == mlp, ignored for plain
    parser.add_argument("--embedding_dim", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)

    parser.add_argument(
        "--initialization", type=str, default="random", choices=["random", "zeros"]
    )
    args = parser.parse_args()

    dataset = PointCloudDataset(args.path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=25000, shuffle=True)

    if args.model == "plain":
        model = NeuralPoissonPlain(
            args.num_embeddings,
            args.grid_dim,
            args.sparse_grid_dim,
            initialization="zeros",
            device=torch.device("cuda:0"),
        )

    elif args.model == "mlp":
        model = NeuralPoissonMLP(
            args.num_embeddings,
            args.embedding_dim,
            args.grid_dim,
            args.sparse_grid_dim,
            args.initialization,
            args.num_layers,
            args.hidden_dim,
            device=torch.device("cuda:0"),
        )

    else:
        model = SirenMLP(in_dim=3, out_dim=1, hidden_dim=256, num_layers=5,
                         device=torch.device("cuda:0"))

    print(model)

    # activate sparse grids
    model.spatial_init_(torch.from_numpy(dataset.positions).cuda())

    # o3d.visualization.draw(
    #     [dataset.pcd, model.bbox_lineset, model.visualize_occupied_cells()]
    # )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model.marching_cubes("mesh_init.ply")

    for epoch in range(20):
        pbar = tqdm(dataloader)

        for batch in pbar:
            optimizer.zero_grad()

            positions = batch["position"].cuda()
            normals = batch["normal"].cuda()

            sdf, grad_x, mask = model(positions)

            loss_surface = sdf[mask].pow(2).mean()

            # TODO: check how SIREN implements this
            loss_surface_normal = (grad_x - normals)[mask].pow(2).sum(dim=-1).mean() + (
                1 - F.cosine_similarity(grad_x[mask], normals[mask], dim=-1)
            ).mean()

            # loss_surface_normal = (grad_x - normals)[mask].pow(2).sum(dim=-1).mean() + (
            #     1 - (grad_x * normals)[mask].sum(dim=1)
            # ).pow(2).mean()
            loss_surface_eikonal = (torch.norm(grad_x[mask], dim=-1) - 1).pow(2).mean()

            # TODO: this sampling is very aggressive and are only in the active grids
            # Ablation in SIREN to see if this causes the problem
            rand_positions = model.sample(num_samples=int(len(positions)))
            sdf_rand, grad_x_rand, mask_rand = model(rand_positions)

            # TODO: this is kind of neural poisson but not identical
            loss_rand_sdf = torch.exp(- 1e2 * sdf_rand[mask_rand].abs()).mean()
            loss_rand_eikonal = (
                (torch.norm(grad_x_rand[mask_rand], dim=-1) - 1).pow(2).mean()
            )

            loss = (
                loss_surface * 3e3
                + loss_surface_normal * 1e3
                + loss_surface_eikonal * 5
                + loss_rand_sdf * 1e2
                + loss_rand_eikonal * 5
            )
            loss.backward()

            pbar.set_description(
                f"Total={loss.item():.4f},"
                f"sdf={loss_surface.item():.4f},"
                f"normal={loss_surface_normal.item():.4f},"
                f"eikonal={loss_surface_eikonal.item():.4f},"
                f"rand_sdf={loss_rand_sdf.item():.4f},"
                f"rand_eikonal={loss_rand_eikonal.item():.4f}"
            )
            optimizer.step()

        # scheduler.step()
        model.marching_cubes(f"mesh_{epoch:03d}.ply")
