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

from rgbd_fusion import TSDFFusion


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--depth_type", type=str, default="sensor", choices=["sensor", "learned"])
    parser.add_argument("--depth_max", type=float, default=4.0, help="max depth value to truncate in meters")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="voxel size in meters in the metric space")
    parser.add_argument("--normalize_scene", action="store_true", help="Normalize scene into the [0, 1] bounding box")
    args = parser.parse_args()
    # fmt: on

    # Load data
    dataset = ImageDataset(
        args.path,
        depth_type=args.depth_type,
        depth_max=args.depth_max,
        normalize_scene=args.normalize_scene,
        image_only=False,
    )
    fuser = TSDFFusion(voxel_size=args.voxel_size, normalize_scene=args.normalize_scene)
    fuser.fuse_dataset(dataset)
    print(f"hash map size after fusion: {fuser.grid.engine.size()}")
    fuser.prune_(0.5)
    print(f"hash map size after pruning: {fuser.grid.engine.size()}")

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

    def sigma_fn(x):
        embeddings, masks = fuser.grid(x, interpolation="linear")
        sdfs = embeddings[..., 0].contiguous()

        beta = 0.01
        alpha = 1.0 / beta
        sigmas = (0.5 * alpha) * (1.0 + sdfs.sign() * torch.expm1(-sdfs.abs() / beta))
        sigmas = torch.where(masks, sigmas, torch.zeros_like(sigmas))
        return sigmas.view(-1, 1)

    def rgb_sigma_fn(t_nears, t_fars, ray_indices):
        print(t_nears.shape, t_fars.shape, ray_indices.shape)
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
    o3d.visualization.draw([mesh])

    batch_size = 2048
    pixel_count = dataset.H * dataset.W
    batches_per_image = pixel_count // batch_size
    assert pixel_count % batch_size == 0
    dataloader = Dataloader(
        dataset, batch_size=batch_size, shuffle=False, device=torch.device("cuda:0")
    )

    for i in range(dataset.num_images):
        colors = []
        im_weights = []
        im_depths = []
        im_normals = []
        for b in range(batches_per_image):
            datum = next(iter(dataloader))
            rays_o = datum["rays_o"]
            rays_d = datum["rays_d"]
            ray_norms = datum["rays_d_norm"]

            # Ray sampling
            (
                ray_indices,
                t_nears,
                t_fars,
                prefix_sum_ray_samples,
            ) = fuser.grid.ray_sample(
                rays_o=rays_o,
                rays_d=rays_d,
                t_min=0.1,
                t_max=1.4,
                t_step=0.01,
            )

            # Sample positions
            x = (
                rays_o[ray_indices]
                + 0.5 * (t_nears + t_fars).view(-1, 1) * rays_d[ray_indices]
            )
            sample_pcd = o3d.t.geometry.PointCloud(x.cpu().numpy())

            sample_weights = weight_fn(x)
            # sample_masks = torch.ones_like(
            #     sample_weights, dtype=bool
            # ).squeeze()
            sample_masks = (sample_weights >= 1.0).squeeze()

            masked_ray_indices = ray_indices[sample_masks]
            sum_masked_ray_samples = torch.zeros(
                (len(rays_o),), dtype=ray_indices.dtype, device=torch.device("cuda:0")
            )
            sum_masked_ray_samples.index_add_(
                0, masked_ray_indices, torch.ones_like(masked_ray_indices)
            )

            sigmas = sigma_fn(x)
            weights = nerfacc.render_weight_from_density(
                t_starts=t_nears[sample_masks].view(-1, 1),
                t_ends=t_fars[sample_masks].view(-1, 1),
                sigmas=sigmas[sample_masks],
                ray_indices=ray_indices[sample_masks],
                n_rays=len(rays_o),
            )
            print(weights.shape)

            rgbs = color_fn(x)
            color = nerfacc.accumulate_along_rays(
                weights, ray_indices[sample_masks], values=rgbs[sample_masks], n_rays=len(rays_o)
            )

            normals = normal_fn(x)
            normal = nerfacc.accumulate_along_rays(
                weights, ray_indices[sample_masks], values=normals[sample_masks], n_rays=len(rays_o)
            )
            sum_weights = nerfacc.accumulate_along_rays(
                weights, ray_indices[sample_masks], values=None, n_rays=len(rays_o)
            )

            depth = nerfacc.accumulate_along_rays(
                weights,
                ray_indices[sample_masks],
                values=0.5 * (t_nears[sample_masks] + t_fars[sample_masks]).view(-1, 1),
                n_rays=len(rays_o),
            )
            depth = depth / ray_norms.view(-1, 1)

            colors.append(color.detach().cpu().numpy())
            im_weights.append(sum_weights.detach().cpu().numpy())
            im_depths.append(depth.detach().cpu().numpy())
            im_normals.append(normal.detach().cpu().numpy())
        colors = np.stack((colors), axis=0).reshape((dataset.H, dataset.W, 3))
        im_weights = np.stack((im_weights), axis=0).reshape((dataset.H, dataset.W, 1))
        im_depths = np.stack((im_depths), axis=0).reshape((dataset.H, dataset.W, 1))
        im_normals = np.stack((im_normals), axis=0).reshape((dataset.H, dataset.W, 3))
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(colors / (im_weights + 1e-6))
        axs[0, 1].imshow(im_weights)
        axs[1, 0].imshow(im_depths)
        axs[1, 1].imshow(im_normals)
        plt.show()
