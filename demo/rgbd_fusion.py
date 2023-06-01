from typing import Union
import torch
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

from data_provider import ImageDataset, Dataloader


def to_o3d(tensor):
    return o3c.Tensor.from_dlpack(to_dlpack(tensor))


def from_o3d(tensor):
    return from_dlpack(tensor.to_dlpack())


def to_o3d_im(tensor):
    return o3d.t.geometry.Image(to_o3d(tensor))


class TSDFFusion:
    """Use ASH's hashgrid to generate differentiable sparse-dense grid from RGB + scaled monocular depth prior"""

    device = torch.device("cuda:0")

    def __init__(
        self,
        grid: Union[UnBoundedSparseDenseGrid, BoundedSparseDenseGrid],
    ):
        self.grid = grid
        if isinstance(self.grid, BoundedSparseDenseGrid):
            self.voxel_size = self.grid.cell_size.min().item()
        else:
            self.voxel_size = self.grid.cell_size
        self.trunc = 2 * self.voxel_size * self.grid.grid_dim

    @torch.no_grad()
    def fuse_dataset(self, dataset):
        pbar = tqdm(range(dataset.num_images))
        for i in pbar:
            pbar.set_description(f"Fuse frame {i}")
            datum = DotDict(dataset.get_image(i))
            for k, v in datum.items():
                if isinstance(v, np.ndarray):
                    datum[k] = torch.from_numpy(v.astype(np.float32)).to(self.device)
            self.fuse_frame(datum)

    @torch.no_grad()
    def unproject_depth_to_points(
        self, depth, intrinsic, extrinsic, depth_scale, depth_max
    ):
        # Multiply back to make open3d happy
        depth_im = to_o3d_im(depth * depth_scale)
        pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth_im,
            to_o3d(intrinsic.contiguous().double()),
            to_o3d(extrinsic.contiguous().double()),
            depth_scale,
            depth_max,
        )
        if len(pcd.point["positions"]) == 0:
            warnings.warn("No points after unprojection")
            return None
        points = from_o3d(pcd.point["positions"])
        return points

    @torch.no_grad()
    def project_points_to_rgbd(
        self, points, intrinsic, extrinsic, color, depth, depth_max
    ):
        h, w, _ = color.shape
        xyz = points @ extrinsic[:3, :3].t() + extrinsic[:3, 3:].t()
        uvd = xyz @ intrinsic.t()

        # In bound validity
        d = uvd[:, 2]
        u = (uvd[:, 0] / uvd[:, 2]).round().long()
        v = (uvd[:, 1] / uvd[:, 2]).round().long()

        mask_projection = (d > 0) * (u >= 0) * (v >= 0) * (u < w) * (v < h)

        u_valid = u[mask_projection]
        v_valid = v[mask_projection]

        depth_readings = torch.zeros_like(d)
        depth_readings[mask_projection] = depth[v_valid, u_valid]

        color_readings = torch.zeros((len(d), 3), device=self.device)
        color_readings[mask_projection] = color[v_valid, u_valid, :]

        sdf = depth_readings - d
        rgb = color_readings

        mask_depth = (
            (depth_readings > 0) * (depth_readings < depth_max) * (sdf >= -self.trunc)
        )
        sdf[sdf >= self.trunc] = self.trunc

        weight = (mask_depth * mask_projection).float()

        return sdf, rgb, weight

    @torch.no_grad()
    def prune_(self, grid_mean_weight_thr=1):
        grid_coords, cell_coords, grid_indices, cell_indices = self.grid.items()

        batch_size = min(1000, len(grid_coords))

        for i in range(0, len(grid_coords), batch_size):
            grid_coords_batch = grid_coords[i : i + batch_size]
            grid_indices_batch = grid_indices[i : i + batch_size]

            weight = self.grid.embeddings[grid_indices_batch, cell_indices, 4]
            mask = weight.mean(dim=1) < grid_mean_weight_thr

            if mask.sum() > 0:
                self.grid.engine.erase(grid_coords_batch[mask].squeeze(1))
                self.grid.embeddings[grid_indices_batch[mask]] = 0
        self.grid.construct_sparse_neighbor_tables_()

    @torch.no_grad()
    def fuse_frame(self, datum):
        torch.cuda.empty_cache()
        datum.depth *= datum.depth_scale

        points = self.unproject_depth_to_points(
            datum.depth,
            datum.intrinsic,
            datum.extrinsic,
            1.0 / datum.depth_scale,  # open3d uses inverse depth scale
            datum.depth_max,
        )
        if points is None:
            return

        # Insertion
        (
            grid_coords,
            cell_coords,
            grid_indices,
            cell_indices,
        ) = self.grid.spatial_init_(points, dilation=2, bidirectional=True)
        if len(grid_indices) == 0:
            return

        cell_positions = self.grid.cell_to_world(grid_coords, cell_coords)

        # Observation
        sdf, rgb, w = self.project_points_to_rgbd(
            cell_positions,
            datum.intrinsic,
            datum.extrinsic,
            datum.rgb,
            datum.depth,
            datum.depth_max,
        )

        # Fusion
        embedding = self.grid.embeddings[grid_indices, cell_indices]

        w_sum = embedding[..., 4:5]
        sdf_mean = embedding[..., 0:1]
        rgb_mean = embedding[..., 1:4]

        w = w.view(w_sum.shape)
        sdf = sdf.view(sdf_mean.shape)
        rgb = rgb.view(rgb_mean.shape)

        w_updated = w_sum + w
        sdf_updated = (sdf_mean * w_sum + sdf * w) / (w_updated + 1e-6)
        rgb_updated = (rgb_mean * w_sum + rgb * w) / (w_updated + 1e-6)

        embedding[..., 4:5] = w_updated
        embedding[..., 0:1] = sdf_updated
        embedding[..., 1:4] = rgb_updated
        self.grid.embeddings[grid_indices, cell_indices] = embedding


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--voxel_size", type=float, default=-1)
    parser.add_argument("--resolution", type=int, default=256)

    parser.add_argument("--depth_type", type=str, default="sensor", choices=["sensor", "learned"])
    parser.add_argument("--depth_max", type=float, default=5.0, help="max depth value to truncate in meters")
    args = parser.parse_args()
    # fmt: on

    device = torch.device("cuda:0")

    if args.voxel_size > 0:
        print(
            f"Using metric voxel size {args.voxel_size}m with UnboundedSparseDenseGrid."
        )
        normalize_scene = False
        grid = UnBoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=80000,
            embedding_dim=5,
            grid_dim=8,
            cell_size=args.voxel_size,
            device=device,
        )

    else:
        print(f"Using resolution {args.resolution} with BoundedSparseDenseGrid.")
        normalize_scene = True
        grid = BoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=80000,
            embedding_dim=5,
            grid_dim=8,
            sparse_grid_dim=32,
            bbox_min=-1 * torch.ones(3, device=device),
            bbox_max=torch.ones(3, device=device),
            device=device,
        )

    # Load data
    dataset = ImageDataset(
        args.path,
        depth_type=args.depth_type,
        depth_max=args.depth_max,
        normalize_scene=normalize_scene,
        image_only=False,
    )

    fuser = TSDFFusion(grid)
    fuser.fuse_dataset(dataset)
    print(f"hash map size after fusion: {fuser.grid.engine.size()}")

    bbox_min, bbox_max = fuser.grid.get_bbox()
    bbox_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox_min.cpu().numpy(), max_bound=bbox_max.cpu().numpy()
        )
    )

    dataloader = Dataloader(
        dataset, batch_size=100, shuffle=True, device=torch.device("cuda:0")
    )
    datum = next(iter(dataloader))

    rays_o = datum["rays_o"]
    rays_d = datum["rays_d"]
    t_min = 0.1
    t_max = 1.7 if normalize_scene else args.depth_max
    ray_indices, t_nears, t_fars, prefix_sum_ray_samples = fuser.grid.ray_sample(
        rays_o=rays_o,
        rays_d=rays_d,
        t_min=t_min,
        t_max=t_max,
        t_step=0.02,
    )

    lineset = o3d.t.geometry.LineSet()
    positions = torch.cat([rays_o + t_min * rays_d, rays_o + t_max * rays_d], dim=0)
    indices = torch.cat(
        [
            torch.arange(len(rays_o)).view(-1, 1),
            torch.arange(len(rays_o), 2 * len(rays_o)).view(-1, 1),
        ],
        dim=-1,
    )
    lineset.point.positions = positions.cpu().numpy()
    lineset.line.indices = indices.cpu().numpy().astype(np.int32)

    sample_positions = (
        rays_o[ray_indices] + 0.5 * (t_nears + t_fars).view(-1, 1) * rays_d[ray_indices]
    )
    sample_pcd = o3d.t.geometry.PointCloud(sample_positions.cpu().numpy())

    # sdf_fn and weight_fn
    def color_fn(x):
        embeddings, masks = fuser.grid(x, interpolation="linear")
        return embeddings[..., 1:4].contiguous()

    def weight_fn(x):
        embeddings, masks = fuser.grid(x, interpolation="linear")
        return embeddings[..., 4:5].contiguous()

    def grad_fn(x):
        x.requires_grad_(True)
        embeddings, masks = fuser.grid(x, interpolation="linear")

        grad_x = torch.autograd.grad(
            outputs=embeddings[..., 0],
            inputs=x,
            grad_outputs=torch.ones_like(embeddings[..., 0], requires_grad=False),
            create_graph=True,
        )[0]
        return grad_x

    def normal_fn(x):
        return F.normalize(grad_fn(x), dim=-1).contiguous()

    def rgb_sigma_fn(t_nears, t_fars, ray_indices):
        positions = rays_o[ray_indices] + rays_d[ray_indices] * (
            0.5 * (t_nears + t_fars)
        )

        embeddings, masks = fuser.grid(positions, interpolation="linear")

        sdfs = embeddings[..., 0].contiguous()
        rgbs = embeddings[..., 1:4].contiguous()

        beta = 0.01
        alpha = 1.0 / beta
        sigmas = (0.5 * alpha) * (1.0 + sdfs.sign() * torch.expm1(-sdfs.abs() / beta))
        sigmas = torch.where(masks, sigmas, torch.zeros_like(sigmas))
        return rgbs, sigmas.view(-1, 1)

    sample_weights = weight_fn(sample_positions)
    mask = (sample_weights >= 1.0).squeeze()

    masked_ray_indices = ray_indices[mask]
    sum_masked_ray_samples = torch.zeros(
        (len(rays_o),), dtype=ray_indices.dtype, device=torch.device("cuda:0")
    )
    sum_masked_ray_samples.index_add_(
        0, masked_ray_indices, torch.ones_like(masked_ray_indices)
    )
    color, opacity, depth = nerfacc.rendering(
        t_nears[mask].view(-1, 1),
        t_fars[mask].view(-1, 1),
        ray_indices[mask],
        n_rays=len(rays_o),
        rgb_sigma_fn=rgb_sigma_fn,
    )

    lineset.line.colors = color.detach().cpu().numpy()

    sample_colors = color_fn(sample_positions)
    sample_normals = normal_fn(sample_positions)

    sample_pcd.point.positions = sample_positions[mask].detach().cpu().numpy()
    sample_pcd.point.colors = sample_colors[mask].detach().cpu().numpy()
    sample_pcd.point.normals = sample_normals[mask].detach().cpu().numpy()

    sdf = fuser.grid.embeddings[..., 0].contiguous()
    weight = fuser.grid.embeddings[..., 4].contiguous()
    mesh = fuser.grid.marching_cubes(
        sdf,
        weight,
        vertices_only=False,
        color_fn=color_fn,
        normal_fn=normal_fn,
        iso_value=0.0,
        weight_thr=0.5,
    )
    o3d.visualization.draw([mesh, lineset, sample_pcd, bbox_lineset])
