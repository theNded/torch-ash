import torch
import torch.nn as nn
import torch.nn.functional as F

import nerfacc

from ash import UnBoundedSparseDenseGrid
import numpy as np
from tqdm import tqdm

import open3d as o3d

from data_provider import ImageDataset, Dataloader

from rgbd_fusion import TSDFFusion
from depth_loss import ScaleAndShiftInvariantLoss


class SDFToSigma(nn.Module):
    def __init__(self, beta, device):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([beta], device=device))

    def forward(self, sdf):
        beta = torch.clamp(self.beta, min=1e-4)

        alpha = 1.0 / beta
        sigma = (0.5 * alpha) * (1.0 + sdf.sign() * torch.expm1(-sdf.abs() / beta))
        return sigma


class PlainVoxels(nn.Module):
    def __init__(self, voxel_size, device):
        super().__init__()
        print(f'voxel_size={voxel_size}')
        self.grid = UnBoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=40000,
            embedding_dim=5,
            grid_dim=8,
            cell_size=voxel_size,
            device=device,
        )

        self.sdf_to_sigma = SDFToSigma(beta=voxel_size, device=device)

    def fuse_dataset(self, dataset, dilation):
        fuser = TSDFFusion(self.grid, dilation=dilation)
        fuser.fuse_dataset(dataset)
        fuser.prune_(0.5)

        print(f"hash map size after pruning: {self.grid.engine.size()}")

    def forward(self, rays_o, rays_d, rays_d_norm, near, far):
        (ray_indices, t_nears, t_fars, prefix_sum_ray_samples,) = self.grid.ray_sample(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=near,
            t_max=far,
            t_step=0.01,
        )

        t_nears = t_nears.view(-1, 1)
        t_fars = t_fars.view(-1, 1)
        t_mid = 0.5 * (t_nears + t_fars)
        x = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        x.requires_grad_(True)
        embeddings, masks = self.grid(x, interpolation="linear")

        # Used for filtering out empty voxels
        voxel_weights = embeddings[..., 4:5]
        masks = masks & (voxel_weights >= 1.0).squeeze()

        valid_ray_indices = ray_indices[masks]
        valid_t_nears = t_nears[masks]
        valid_t_fars = t_fars[masks]
        valid_t_mids = t_mid[masks]

        # Could optimize a bit
        embeddings = embeddings[masks]
        sdfs = embeddings[..., 0:1].contiguous()
        rgbs = embeddings[..., 1:4].contiguous()
        sdf_grads = torch.autograd.grad(
            outputs=sdfs,
            inputs=x,
            grad_outputs=torch.ones_like(sdfs, requires_grad=False),
            create_graph=True,
        )[0]
        sdf_grads = sdf_grads[masks]
        normals = F.normalize(sdf_grads, dim=-1)
        sigmas = self.sdf_to_sigma(sdfs)

        weights = nerfacc.render_weight_from_density(
            t_starts=valid_t_nears,
            t_ends=valid_t_fars,
            sigmas=sigmas,
            ray_indices=valid_ray_indices,
            n_rays=len(rays_o),
        )

        # TODO: could use concatenated rendering in one pass and dispatch
        # TODO: also can reuse the packed info
        rendered_rgb = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=valid_ray_indices,
            values=rgbs,
            n_rays=len(rays_o),
        )

        rendered_depth = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=valid_ray_indices,
            values=valid_t_mids,
            n_rays=len(rays_o),
        )
        rendered_depth = rendered_depth / rays_d_norm

        rendered_normals = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=valid_ray_indices,
            values=normals,
            n_rays=len(rays_o),
        )

        accumulated_weights = nerfacc.accumulate_along_rays(
            weights=weights,
            ray_indices=valid_ray_indices,
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

    def marching_cubes(self):
        sdf = self.grid.embeddings[..., 0].contiguous()
        weight = self.grid.embeddings[..., 4].contiguous()
        mesh = self.grid.marching_cubes(
            sdf,
            weight,
            vertices_only=False,
            color_fn=self.color_fn,
            normal_fn=self.normal_fn,
            iso_value=0.0,
            weight_thr=0.5,
        )
        return mesh

    def color_fn(self, x):
        embeddings, masks = self.grid(x, interpolation="linear")
        return embeddings[..., 1:4].contiguous()

    def weight_fn(self, x):
        embeddings, masks = self.grid(x, interpolation="linear")
        return embeddings[..., 4:5].contiguous()

    def grad_fn(self, x):
        x.requires_grad_(True)
        embeddings, masks = self.grid(x, interpolation="linear")

        grad_x = torch.autograd.grad(
            outputs=embeddings[..., 0],
            inputs=x,
            grad_outputs=torch.ones_like(embeddings[..., 0], requires_grad=False),
            create_graph=True,
        )[0]
        return grad_x

    def normal_fn(self, x):
        return F.normalize(self.grad_fn(x), dim=-1).contiguous()


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--voxel_size", type=float, default=0.015)
    parser.add_argument("--depth_type", type=str, default="sensor", choices=["sensor", "learned"])
    parser.add_argument("--depth_max", type=float, default=5.0, help="max depth value to truncate in meters")
    args = parser.parse_args()
    # fmt: on

    # Load data
    dataset = ImageDataset(
        args.path,
        depth_type=args.depth_type,
        depth_max=args.depth_max,
        normalize_scene=False,
        generate_rays=True,
    )

    model = PlainVoxels(voxel_size=args.voxel_size, device=torch.device("cuda:0"))
    dilation = 2
    model.fuse_dataset(dataset, dilation)
    model.grid.gaussian_filter_(7, 0.1)
    mesh = model.marching_cubes()
    o3d.visualization.draw([mesh])

    batch_size = 2048
    pixel_count = dataset.H * dataset.W
    batches_per_image = pixel_count // batch_size
    assert pixel_count % batch_size == 0

    # eval_dataloader = Dataloader(
    #     dataset, batch_size=batch_size, shuffle=False, device=torch.device("cuda:0")
    # )

    train_dataloader = Dataloader(
        dataset, batch_size=batch_size, shuffle=True, device=torch.device("cuda:0")
    )

    optim = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9999)

    # Training
    depth_loss_fn = ScaleAndShiftInvariantLoss()
    pbar = tqdm(range(5001))
    for step in pbar:
        optim.zero_grad()
        datum = next(iter(train_dataloader))
        rays_o = datum["rays_o"]
        rays_d = datum["rays_d"]
        ray_norms = datum["rays_d_norm"]
        result = model(rays_o, rays_d, ray_norms, near=0.1, far=4.0)

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

        if step % 500 == 0 and step > 0:
            mesh = model.marching_cubes()
            o3d.visualization.draw([mesh])
            scheduler.step()

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

        im_rgbs = np.concatenate(im_rgbs, axis=0).reshape(dataset.H, dataset.W, 3)
        im_weights = np.concatenate(im_weights, axis=0).reshape(dataset.H, dataset.W, 1)
        im_depths = np.concatenate(im_depths, axis=0).reshape(dataset.H, dataset.W, 1)
        im_normals = np.concatenate(im_normals, axis=0).reshape(dataset.H, dataset.W, 3)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(im_rgbs)
        axes[0, 1].imshow(im_weights)
        axes[1, 0].imshow(im_depths)
        axes[1, 1].imshow(im_normals)
        plt.show()
