import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack

import nerfacc

from ash import UnBoundedSparseDenseGrid, BoundedSparseDenseGrid, DotDict
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

import open3d as o3d
import open3d.core as o3c

from ash import UnBoundedSparseDenseGrid, BoundedSparseDenseGrid, DotDict, MLP
from data_provider import ImageDataset, Dataloader

from rgbd_fusion import TSDFFusion
from depth_loss import ScaleAndShiftInvariantLoss
from mesh_util import create_mesh


class SDFToSigma(nn.Module):
    def __init__(self, beta, device):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta], device=device))

    def forward(self, sdf):
        beta = torch.clamp(self.beta, min=1e-4)

        alpha = 1.0 / beta
        sigma = (0.5 * alpha) * (1.0 + sdf.sign() * torch.expm1(-sdf.abs() / beta))
        return sigma


class MonoSDF(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device
        self.grid_dim = 4
        self.sparse_grid_dim = 32

        self.grid = BoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=self.sparse_grid_dim**3,
            embedding_dim=8,
            grid_dim=self.grid_dim,
            sparse_grid_dim=self.sparse_grid_dim,
            bbox_min=-1 * torch.ones(3, device=device),
            bbox_max=torch.ones(3, device=device),
            device=device,
        )
        nn.init.uniform_(self.grid.embeddings, -1e-4, 1e-4)

        self.geometry_mlp = MLP(
            dim_in=3 + 8,
            dim_hidden=256,
            dim_out=1 + 13,
            num_layers=3,
            radius=0.8,
            sphere_init=True,
            inside_out=True,
        ).to(device)

        self.color_mlp = MLP(
            dim_in=3 + 13 + 3,
            dim_hidden=256,
            dim_out=3,
            num_layers=3,
            final_activation="sigmoid",
        ).to(device)

        self.sdf_to_sigma = SDFToSigma(beta=0.01, device=device)

    def fuse_dataset(self, dataset):
        self.tsdf_grid = BoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=60000,
            embedding_dim=5,
            grid_dim=self.grid_dim,
            sparse_grid_dim=self.sparse_grid_dim,
            bbox_min=-1 * torch.ones(3, device=self.device),
            bbox_max=torch.ones(3, device=self.device),
            device=self.device,
        )

        fuser = TSDFFusion(self.tsdf_grid)
        fuser.fuse_dataset(dataset)
        print(f"hash map size before pruning: {self.tsdf_grid.engine.size()}")
        fuser.prune_(3)
        print(f"hash map size after pruning: {self.tsdf_grid.engine.size()}")

        self.grid.engine.insert_keys(self.tsdf_grid.engine.keys())
        # self.grid.dense_init_()

    def forward(self, rays_o, rays_d, rays_d_norm, near, far):
        (
            ray_indices,
            t_nears,
            t_fars,
            prefix_sum_ray_samples,
        ) = self.tsdf_grid.ray_sample(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=near,
            t_max=far,
            t_step=0.01,
        )

        t_nears = t_nears.view(-1, 1)
        t_fars = t_fars.view(-1, 1)
        t_mids = 0.5 * (t_nears + t_fars)
        x = rays_o[ray_indices] + t_mids * rays_d[ray_indices]

        x.requires_grad_(True)
        embeddings, masks = self.grid(x, interpolation="linear")

        # Use zeros for empty space
        # TODO: empty space grid
        embeddings = torch.where(
            masks.view(-1, 1), embeddings, torch.zeros_like(embeddings)
        )

        geometry_output = self.geometry_mlp(torch.cat([x, embeddings], dim=-1))
        sdfs, geom_embeddings = torch.split(geometry_output, [1, 13], dim=-1)
        sdf_grads = torch.autograd.grad(
            outputs=sdfs,
            inputs=x,
            grad_outputs=torch.ones_like(sdfs, requires_grad=False),
            create_graph=True,
        )[0]
        normals = F.normalize(sdf_grads, dim=-1)

        rgbs = self.color_mlp(
            torch.cat(
                [
                    rays_d[ray_indices],
                    geom_embeddings,
                    normals,
                ],
                dim=-1,
            )
        )

        sigmas = self.sdf_to_sigma(sdfs)

        weights = nerfacc.render_weight_from_density(
            t_starts=t_nears,
            t_ends=t_fars,
            sigmas=sigmas,
            ray_indices=ray_indices,
            n_rays=len(rays_o),
        )

        # TODO: could use concatenated rendering in one pass and dispatch
        # TODO: also can reuse the packed info
        rendered_rgb = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=rgbs,
            n_rays=len(rays_o),
        )

        rendered_depth = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=t_mids,
            n_rays=len(rays_o),
        )
        rendered_depth = rendered_depth / rays_d_norm

        rendered_normals = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=normals,
            n_rays=len(rays_o),
        )

        accumulated_weights = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=ray_indices,
            values=None,
            n_rays=len(rays_o),
        )

        return {
            "rgb": rendered_rgb,
            "depth": rendered_depth,
            "normal": rendered_normals,
            "weights": accumulated_weights,
            "sdf_grads": sdf_grads,
        }

    @torch.no_grad()
    def visualize_occupied_cells(self):
        grid_coords, cell_coords, grid_indices, cell_indices = self.grid.items()
        cell_positions = self.grid.cell_to_world(grid_coords, cell_coords)
        pcd = o3d.t.geometry.PointCloud(cell_positions.cpu().numpy())
        pcd.paint_uniform_color([0.0, 0.0, 1.0])
        return pcd

    def marching_cubes(self):
        # grid_coords, cell_coords, _, _ = self.tsdf_grid.items()
        # cell_positions = self.tsdf_grid.cell_to_world(grid_coords, cell_coords)
        # sdf = self.sdf_fn(cell_positions).detach().contiguous()
        sdf = self.tsdf_grid.embeddings[..., 0].contiguous()
        weight = self.tsdf_grid.embeddings[..., 4].contiguous()
        # print(sdf.shape, weight.shape)
        mesh = self.tsdf_grid.marching_cubes(
            sdf,
            weight,
            vertices_only=False,
            color_fn=None,
            normal_fn=None,
            iso_value=0.0,
            weight_thr=0.5,
        )
        return mesh

    def sdf_fn(self, x):
        embeddings, masks = self.grid(x, interpolation="linear")

        # Use zeros for empty space
        # TODO: empty space grid
        embeddings = torch.where(
            masks.view(-1, 1), embeddings, torch.zeros_like(embeddings)
        )

        geometry_output = self.geometry_mlp(torch.cat([x, embeddings], dim=-1))
        sdfs, geom_embeddings = torch.split(geometry_output, [1, 13], dim=-1)

        return sdfs


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--depth_type", type=str, default="sensor", choices=["sensor", "learned"])
    parser.add_argument("--depth_max", type=float, default=4.0, help="max depth value to truncate in meters")
    args = parser.parse_args()
    # fmt: on

    # Load data
    dataset = ImageDataset(
        args.path,
        depth_type=args.depth_type,
        depth_max=args.depth_max,
        normalize_scene=True,
        image_only=False,
    )

    model = MonoSDF(device=torch.device("cuda:0"))
    model.fuse_dataset(dataset)
    mesh = model.marching_cubes()
    mesh.compute_vertex_normals()
    pcd = model.visualize_occupied_cells()
    o3d.visualization.draw([pcd, mesh])

    batch_size = 2048
    pixel_count = dataset.H * dataset.W
    batches_per_image = pixel_count // batch_size
    assert pixel_count % batch_size == 0

    eval_dataloader = Dataloader(
        dataset, batch_size=batch_size, shuffle=False, device=torch.device("cuda:0")
    )

    train_dataloader = Dataloader(
        dataset, batch_size=batch_size, shuffle=True, device=torch.device("cuda:0")
    )

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)


    # Training
    depth_loss_fn = ScaleAndShiftInvariantLoss()
    pbar = tqdm(range(50001))
    for step in pbar:
        optim.zero_grad()
        datum = next(iter(train_dataloader))
        rays_o = datum["rays_o"]
        rays_d = datum["rays_d"]
        ray_norms = datum["rays_d_norm"]
        result = model(rays_o, rays_d, ray_norms, near=0.1, far=1.4)

        rgb_gt = datum["rgb"]
        normal_gt = datum["normal"]
        depth_gt = datum["depth"]

        rgb_loss = F.mse_loss(result["rgb"], rgb_gt)
        normal_loss = (
            1 - F.cosine_similarity(result["normal"], normal_gt, dim=-1).mean()
        )
        depth_loss = depth_loss_fn(
            result["depth"].view(-1, 32, 32),
            depth_gt.view(-1, 32, 32),
            torch.ones_like(depth_gt.view(-1, 32, 32)).bool(),
        )

        eikonal_loss = (torch.norm(result["sdf_grads"], dim=-1) - 1).abs().mean()

        loss = rgb_loss + 0.1 * normal_loss + 0.1 * depth_loss + 0.1 * eikonal_loss

        loss.backward()
        pbar.set_description(
            f"loss: {loss.item():.4f},"
            f"rgb: {rgb_loss.item():.4f},"
            f"normal: {normal_loss.item():.4f},"
            f"depth: {depth_loss.item():.4f},"
            f"eikonal: {eikonal_loss.item():.4f}",
        )
        optim.step()

        if step % 1000 == 0 and step > 0:
            create_mesh(model.sdf_fn, f"mesh_{step:03d}.ply")

            # Evaluation
            for i in range(dataset.num_images):
                im_rgbs = []
                im_weights = []
                im_depths = []
                im_normals = []

                for b in range(batches_per_image):
                    datum = next(iter(eval_dataloader))
                    rays_o = datum["rays_o"]
                    rays_d = datum["rays_d"]
                    ray_norms = datum["rays_d_norm"]
                    result = model(rays_o, rays_d, ray_norms, near=0.1, far=1.4)

                    im_rgbs.append(result["rgb"].detach().cpu().numpy())
                    im_weights.append(result["weights"].detach().cpu().numpy())
                    im_depths.append(result["depth"].detach().cpu().numpy())
                    im_normals.append(result["normal"].detach().cpu().numpy())

                im_rgbs = np.concatenate(im_rgbs, axis=0).reshape(
                    dataset.H, dataset.W, 3
                )
                im_weights = np.concatenate(im_weights, axis=0).reshape(
                    dataset.H, dataset.W, 1
                )
                im_depths = np.concatenate(im_depths, axis=0).reshape(
                    dataset.H, dataset.W, 1
                )
                im_normals = np.concatenate(im_normals, axis=0).reshape(
                    dataset.H, dataset.W, 3
                )

                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2)
                axes[0, 0].imshow(im_rgbs)
                axes[0, 1].imshow(im_weights)
                axes[1, 0].imshow(im_depths)
                axes[1, 1].imshow(im_normals)
                plt.savefig(f"eval_{step:03d}.png")
                plt.close(fig)
                break
