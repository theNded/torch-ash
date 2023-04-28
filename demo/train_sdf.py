from typing import Tuple

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

from ash import (
    UnBoundedSparseDenseGrid,
    BoundedSparseDenseGrid,
    DotDict,
    BoundedMultiResGrid,
)
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

import open3d as o3d
import open3d.core as o3c

from siren import SirenMLP, SirenNet, create_mesh

import tinycudann as tcnn

from data_provider import PointCloudDataset, Dataloader


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
            nn.init.normal_(self.grid.embeddings, mean=0.0, std=0.001)
            # nn.init.zeros_(self.grid.embeddings)
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
        self.grid.spatial_init_(positions, dilation=0)

    def full_init_(self):
        self.grid.full_init_()

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
        initialization="random",
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


class NeuralPoissonMultiResMLP(nn.Module):
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
        super().__init__()
        self.grid = BoundedMultiResGrid(
            in_dim=3,
            # num_embeddings=[512, 2048, 8196, 32768],
            # embedding_dims=2,
            # grid_dims=2,
            # sparse_grid_dims=[8, 16, 32, 64],
            num_embeddings=[512],
            embedding_dims=2,
            grid_dims=2,
            sparse_grid_dims=[8],
            bbox_min=-1 * torch.ones(3, device=device),
            bbox_max=1 * torch.ones(3, device=device),
            device=device,
        )

        self.mlp = SirenNet(
            dim_in=3 + 2,
            dim_hidden=128,
            dim_out=1,
            num_layers=2,
            w0=30.0,
        ).to(device)

        self.bbox_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
            o3d.geometry.AxisAlignedBoundingBox([-1, -1, -1], [1, 1, 1])
        )


    # @override
    def sample(self, num_samples):

        sample_coords = np.random.uniform(-1, 1, size=(num_samples, 3))
        return torch.from_numpy(sample_coords).float().cuda()

    def visualize_occupied_cells(self):
        pcds = []
        for grid in self.grid.grids:
            grid_coords, cell_coords, grid_indices, cell_indices = grid.items()
            cell_positions = grid.cell_to_world(grid_coords, cell_coords)
            pcd = o3d.t.geometry.PointCloud(cell_positions.cpu().numpy())
            pcd.paint_uniform_color(np.random.rand(3))
            pcds.append(pcd)
        return pcds

    @torch.no_grad()
    def spatial_init_(self, positions):
        self.grid.spatial_init_(positions, dilation=2)

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions.requires_grad_(True)
        embedding, mask = self.grid(positions)
        sdf = self.mlp(torch.cat((embedding, positions), dim=-1))

        grad_x = torch.autograd.grad(
            outputs=sdf[..., 0],
            inputs=positions,
            grad_outputs=torch.ones_like(sdf[..., 0], requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return sdf, grad_x, mask

    def marching_cubes(self, fname):
        def sdf_fn(x):
            sdf, grad_x, mask = self.forward(x)
            return sdf.detach()

        create_mesh(sdf_fn, fname, N=256, max_batch=64**3, offset=None, scale=None)


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

        self.mlp = SirenNet(
            dim_in=3 + 8,
            dim_hidden=128,
            dim_out=1,
            num_layers=2,
            w0=30.0,
        ).to(device)

    # @override
    def sample(self, num_samples):
        sample_coords = np.random.uniform(-1, 1, size=(num_samples, 3))
        return torch.from_numpy(sample_coords).float().cuda()

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions.requires_grad_(True)
        embedding, mask = self.grid(positions, interpolation="linear")

        # embedding = torch.where(
        #     mask.view(-1, 1), embedding, torch.zeros_like(embedding)
        # )

        # sdf = self.mlp(positions)
        sdf = self.mlp(torch.cat((embedding, positions), dim=-1))

        grad_x = torch.autograd.grad(
            outputs=sdf[..., 0],
            inputs=positions,
            grad_outputs=torch.ones_like(sdf[..., 0], requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return sdf, grad_x, mask

    def marching_cubes(self, fname):
        def sdf_fn(x):
            sdf, grad_x, mask = self.forward(x)
            return sdf.detach()

        create_mesh(sdf_fn, fname, N=256, max_batch=64**3, offset=None, scale=None)



class NGP(nn.Module):
    def __init__(
        self,
        num_layers=2,
        hidden_dim=128,
        device=torch.device("cuda:0"),
    ):
        super().__init__()

        self.encoding = tcnn.Encoding(
            3,
            {
                "otype": "DenseGrid",
                "n_levels": 1,
                "n_features_per_level": 8,
                "log2_hashmap_size": 12,
                "base_resolution": 16,
                "per_level_scale": 2,
                "interpolation": "Linear",
            },
        ).to(device)

        self.mlp = SirenNet(
            dim_in=3 + 8,
            dim_hidden=128,
            dim_out=1,
            num_layers=2,
            w0=30.0,
        ).to(device)

    def forward(
        self, positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions.requires_grad_(True)
        embedding = self.encoding(positions)
        sdf = self.mlp(torch.cat((embedding, positions), dim=-1))

        grad_x = torch.autograd.grad(
            outputs=sdf[..., 0],
            inputs=positions,
            grad_outputs=torch.ones_like(sdf[..., 0], requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return sdf, grad_x, torch.ones_like(sdf, dtype=torch.bool).squeeze(-1)

    def sample(self, num_samples):
        sample_coords = np.random.uniform(-1, 1, size=(num_samples, 3))
        return torch.from_numpy(sample_coords).float().cuda()

    def spatial_init_(self, x):
        pass

    def full_init_(self):
        pass

    def marching_cubes(self, fname):
        def sdf_fn(x):
            sdf, grad_x, mask = self.forward(x)
            return sdf.detach()

        create_mesh(sdf_fn, fname, N=256, max_batch=64**3, offset=None, scale=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument(
        "--model", type=str, default="plain", choices=["plain", "mlp", "multires-mlp", "siren", "ngp"]
    )
    parser.add_argument(
        "--grid_dim",
        type=int,
        default=1,
        help="Locally dense grid dimension, e.g. 8 for 8x8x8",
    )
    parser.add_argument(
        "--sparse_grid_dim",
        type=int,
        default=16,
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
    dataloader = Dataloader(dataset, batch_size=25000, shuffle=True)

    if args.model == "plain":
        model = NeuralPoissonPlain(
            args.num_embeddings,
            args.grid_dim,
            args.sparse_grid_dim,
            initialization=args.initialization,
            device=torch.device("cuda:0"),
        )

    elif args.model == "multires-mlp":
        model = NeuralPoissonMultiResMLP(
            args.num_embeddings,
            args.embedding_dim,
            args.grid_dim,
            args.sparse_grid_dim,
            args.initialization,
            args.num_layers,
            args.hidden_dim,
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

    elif args.model == "ngp":
        model = NGP(num_layers=2, hidden_dim=128, device=torch.device("cuda:0"))

    else:
        model = SirenMLP(
            in_dim=3,
            out_dim=1,
            hidden_dim=256,
            num_layers=5,
            device=torch.device("cuda:0"),
        )

    # activate sparse grids
    # model.spatial_init_(torch.from_numpy(dataset.positions).cuda())
    model.full_init_()

    # o3d.visualization.draw(
    #     [dataset.pcd, model.bbox_lineset] + [model.visualize_occupied_cells()])


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model.marching_cubes("mesh_init.ply")

    pbar = tqdm(range(20000))
    for step in pbar:
        batch = next(iter(dataloader))

        optimizer.zero_grad()

        positions = batch["position"].cuda()
        normals = batch["normal"].cuda()

        sdf, grad_x, mask = model(positions)
        #print('surface valid ratio:', mask.sum() / mask.numel())

        loss_surface = sdf[mask].abs().mean()

        loss_surface_normal = (
            1 - F.cosine_similarity(grad_x, normals, dim=-1)
        ).mean()

        # print(sdf.max(), grad_x.max())
        loss_surface_eikonal = (torch.norm(grad_x[mask], dim=-1) - 1).abs().mean()

        # TODO: this sampling is very aggressive and are only in the active grids
        # Ablation in SIREN to see if this causes the problem
        rand_positions = model.sample(num_samples=int(len(positions)))
        sdf_rand, grad_x_rand, mask_rand = model(rand_positions)
        #print(sdf_rand.max(), grad_x_rand.max())

        #print('off-surface valid ratio:', mask_rand.sum() / mask_rand.numel())

        # TODO: this is kind of neural poisson but not identical
        loss_rand_sdf = torch.exp(-1e2 * sdf_rand[mask_rand].abs()).mean()
        loss_rand_eikonal = (
            (torch.norm(grad_x_rand[mask_rand], dim=-1) - 1).abs().mean()
        )

        loss = (
            loss_surface * 0 # 3e3
            + loss_surface_normal * 1e3
            + loss_surface_eikonal * 5e1
            + loss_rand_sdf * 0#1e2
            + loss_rand_eikonal * 0#5e1
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

        if step % 500 == 0:
            scheduler.step()
            model.marching_cubes(f"mesh_{step:03d}.ply")
