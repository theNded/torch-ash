import torch
import torch.nn as nn
import torch.nn.functional as F

import nerfacc

from ash import UnBoundedSparseDenseGrid
import numpy as np
from tqdm import tqdm

import open3d as o3d
from torch.utils.tensorboard import SummaryWriter

from data_provider import ImageDataset, Dataloader

from rgbd_fusion import TSDFFusion
from depth_loss import ScaleAndShiftInvariantLoss


class SDFToDensity(nn.Module):
    def __init__(self, min_beta=0.01, init_beta=1):
        super(SDFToDensity, self).__init__()
        self.min_beta = min_beta
        self.beta = nn.Parameter(torch.Tensor([init_beta]))

    def forward(self, sdf):
        beta = self.min_beta + torch.abs(self.beta)

        alpha = 1 / beta
        # https://github.com/lioryariv/volsdf/blob/main/code/model/density.py
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))


class PlainVoxels(nn.Module):
    def __init__(self, voxel_size, device):
        super().__init__()
        print(f"voxel_size={voxel_size}")
        self.grid = UnBoundedSparseDenseGrid(
            in_dim=3,
            num_embeddings=40000,
            embedding_dim=5,
            grid_dim=8,
            cell_size=voxel_size,
            device=device,
        )

        self.sdf_to_sigma = SDFToDensity(
            min_beta=voxel_size, init_beta=voxel_size * 2
        ).to(device)

    def parameters(self):
        return [
            {"params": self.grid.parameters()},
            {"params": self.sdf_to_sigma.parameters(), "lr": 1e-4},
        ]

    def fuse_dataset(self, dataset, dilation):
        fuser = TSDFFusion(self.grid, dilation=dilation)
        fuser.fuse_dataset(dataset)
        fuser.prune_by_mesh_connected_components_(ratio_to_largest_component=0.5)

        print(f"hash map size after pruning: {self.grid.engine.size()}")

    def forward(self, rays_o, rays_d, rays_d_norm, near, far, jitter=None):
        (rays_near, rays_far) = self.grid.ray_find_near_far(
            rays_o=rays_o,
            rays_d=rays_d,
            t_min=near,
            t_max=far,
            t_step=0.01,  # no use
        )

        (ray_indices, t_nears, t_fars, prefix_sum_ray_samples,) = self.grid.ray_sample(
            rays_o=rays_o,
            rays_d=rays_d,
            rays_near=rays_near,
            rays_far=rays_far,
            max_samples_per_ray=64,
        )
        if jitter is not None:
            t_nears += jitter[..., 0:1]
            t_fars += jitter[..., 1:2]

        t_nears = t_nears.view(-1, 1)
        t_fars = t_fars.view(-1, 1)
        t_mid = 0.5 * (t_nears + t_fars)
        x = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

        x.requires_grad_(True)
        embeddings, masks = self.grid(x, interpolation="linear")
        # import ipdb; ipdb.set_trace()

        # Used for filtering out empty voxels

        # Could optimize a bit
        # embeddings = embeddings[masks]
        masks = masks.view(-1, 1)
        sdfs = embeddings[..., 0:1].contiguous().view(-1, 1)
        rgbs = embeddings[..., 1:4].contiguous().view(-1, 3)
        sdf_grads = torch.autograd.grad(
            outputs=sdfs,
            inputs=x,
            grad_outputs=torch.ones_like(sdfs, requires_grad=False),
            create_graph=True,
        )[0]

        normals = F.normalize(sdf_grads, dim=-1)
        # print(f'normals.shape={normals.shape}, {normals}')
        sigmas = self.sdf_to_sigma(sdfs) * masks.float()

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
            values=t_mid,
            n_rays=len(rays_o),
        )
        rays_near = rays_near.view(-1, 1) / rays_d_norm
        rays_far = rays_far.view(-1, 1) / rays_d_norm
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
        # print(f'rendered_normals.shape={rendered_normals.shape}: {rendered_normals}')

        return {
            "rgb": rendered_rgb,
            "depth": rendered_depth,
            "normal": rendered_normals,
            "weights": accumulated_weights,
            "sdf_grads": sdf_grads,
            "near": rays_near,
            "far": rays_far,
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
            weight_thr=1,
        )
        return mesh

    def occupancy_lineset(self, color=[0, 0, 1], scale=1.0):
        xyz000, _, _, _ = self.grid.items()
        xyz000 = xyz000.view(-1, 3)

        block_len = self.grid.cell_size * self.grid.grid_dim

        xyz000 = xyz000.cpu().numpy() * block_len
        xyz001 = xyz000 + np.array([[block_len * scale, 0, 0]])
        xyz010 = xyz000 + np.array([[0, block_len * scale, 0]])
        xyz100 = xyz000 + np.array([[0, 0, block_len * scale]])
        xyz = np.concatenate((xyz000, xyz001, xyz010, xyz100), axis=0).astype(
            np.float32
        )

        lineset = o3d.t.geometry.LineSet()
        lineset.point["positions"] = o3d.core.Tensor(xyz)

        n = len(xyz000)
        lineset000 = np.arange(0, n)
        lineset001 = np.arange(n, 2 * n)
        lineset010 = np.arange(2 * n, 3 * n)
        lineset100 = np.arange(3 * n, 4 * n)

        indices001 = np.stack((lineset000, lineset001), axis=1)
        indices010 = np.stack((lineset000, lineset010), axis=1)
        indices100 = np.stack((lineset000, lineset100), axis=1)
        indices = np.concatenate((indices001, indices010, indices100), axis=0)

        lineset.line["indices"] = o3d.core.Tensor(indices.astype(np.int32))
        colors = np.tile(color, (3 * n, 1))
        lineset.line["colors"] = o3d.core.Tensor(colors.astype(np.float32))
        return lineset

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
        return grad_x * masks.float().view(-1, 1)

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
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--train_batch_size", type=int, default=65536)
    parser.add_argument("--eval_batch_size", type=int, default=4096)
    args = parser.parse_args()
    # fmt: on

    import datetime

    path = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(path)

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
    model.grid.gaussian_filter_(7, 1)
    mesh = model.marching_cubes()
    # o3d.visualization.draw([mesh, model.occupancy_lineset()])

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size

    pixel_count = dataset.H * dataset.W
    eval_batches_per_image = pixel_count // eval_batch_size
    assert pixel_count % eval_batch_size == 0

    eval_dataloader = Dataloader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        device=torch.device("cuda:0"),
    )

    train_dataloader = Dataloader(
        dataset, batch_size=train_batch_size, shuffle=True, device=torch.device("cuda:0")
    )

    optim = torch.optim.RMSprop(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.9)

    # Training
    depth_loss_fn = ScaleAndShiftInvariantLoss()
    pbar = tqdm(range(args.steps + 1))

    jitter = None

    def reset_jitter_():
        jitter = (
            (torch.rand(train_batch_size, 2, device=torch.device("cuda:0")) * 2 - 1)
            * 0.5
            * args.voxel_size
        )

    for step in pbar:
        optim.zero_grad()
        datum = next(iter(train_dataloader))
        rays_o = datum["rays_o"]
        rays_d = datum["rays_d"]
        ray_norms = datum["rays_d_norm"]
        result = model(rays_o, rays_d, ray_norms, near=0.3, far=5.0, jitter=jitter)

        rgb_gt = datum["rgb"]
        normal_gt = datum["normal"]
        depth_gt = datum["depth"]

        rgb_loss = F.mse_loss(result["rgb"], rgb_gt)
        normal_loss_l1 = F.l1_loss(result["normal"], normal_gt)
        normal_loss_cos = (1 - (result["normal"] * normal_gt).sum(dim=-1)).mean()

        depth_loss = depth_loss_fn(
            result["depth"].view(-1, 32, 32),
            depth_gt.view(-1, 32, 32),
            torch.ones_like(depth_gt.view(-1, 32, 32)).bool(),
        )

        eikonal_loss_ray = (torch.norm(result["sdf_grads"], dim=-1) - 1).abs().mean()

        uniform_samples = model.grid.uniform_sample(train_batch_size)
        uniform_sdf_grads = model.grad_fn(uniform_samples)
        eikonal_loss_uniform = (torch.norm(uniform_sdf_grads, dim=-1) - 1).abs().mean()

        loss = (
            rgb_loss
            + 0.05 * normal_loss_l1
            + 0.05 * normal_loss_cos
            + 0.1 * depth_loss
            + 0.1 * eikonal_loss_ray
            + 0.1 * eikonal_loss_uniform
        )

        loss.backward()
        pbar.set_description(
            f"loss: {loss.item():.4f},"
            f"rgb: {rgb_loss.item():.4f},"
            f"normal_l1: {normal_loss_l1.item():.4f},"
            f"normal_cos: {normal_loss_cos.item():.4f},"
            f"depth: {depth_loss.item():.4f},"
            f"eikonal_ray: {eikonal_loss_ray.item():.4f}",
            f"eikonal_uniform: {eikonal_loss_uniform.item():.4f}",
        )
        optim.step()
        writer.add_scalar("loss/total", loss.item(), step)
        writer.add_scalar("loss/rgb", rgb_loss.item(), step)
        writer.add_scalar("loss/normal_l1", normal_loss_l1.item(), step)
        writer.add_scalar("loss/normal_cos", normal_loss_cos.item(), step)
        writer.add_scalar("loss/depth", depth_loss.item(), step)
        writer.add_scalar("loss/eikonal_ray", eikonal_loss_ray.item(), step)
        writer.add_scalar("loss/eikonal_uniform", eikonal_loss_uniform.item(), step)

        if step % 500 == 0:
            mesh = model.marching_cubes()
            o3d.io.write_triangle_mesh(f"{path}/mesh_{step}.ply", mesh.to_legacy())

            im_rgbs = []
            im_weights = []
            im_depths = []
            im_normals = []
            im_near = []
            im_far = []

            for b in range(eval_batches_per_image):
                datum = next(iter(eval_dataloader))
                rays_o = datum["rays_o"]
                rays_d = datum["rays_d"]
                ray_norms = datum["rays_d_norm"]
                result = model(rays_o, rays_d, ray_norms, near=0.3, far=5.0)

                im_rgbs.append(result["rgb"].detach().cpu().numpy())
                im_weights.append(result["weights"].detach().cpu().numpy())
                im_depths.append(result["depth"].detach().cpu().numpy())
                im_normals.append(result["normal"].detach().cpu().numpy())
                im_near.append(result["near"].detach().cpu().numpy())
                im_far.append(result["far"].detach().cpu().numpy())

            im_rgbs = np.concatenate(im_rgbs, axis=0).reshape(dataset.H, dataset.W, 3)
            im_weights = np.concatenate(im_weights, axis=0).reshape(
                dataset.H, dataset.W, 1
            )
            im_depths = np.concatenate(im_depths, axis=0).reshape(
                dataset.H, dataset.W, 1
            )
            im_normals = (
                np.concatenate(im_normals, axis=0).reshape(dataset.H, dataset.W, 3)
                + 1.0
            ) * 0.5
            im_near = np.concatenate(im_near, axis=0).reshape(dataset.H, dataset.W, 1)
            im_far = np.concatenate(im_far, axis=0).reshape(dataset.H, dataset.W, 1)

            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3)
            axes[0, 0].imshow(im_rgbs)
            axes[0, 1].imshow(im_weights)
            axes[0, 2].imshow(im_near)
            axes[1, 0].imshow((im_depths))
            axes[1, 1].imshow((im_normals))
            axes[1, 2].imshow((im_far))
            for i in range(2):
                for j in range(3):
                    axes[i, j].set_axis_off()
            fig.tight_layout()
            writer.add_figure("eval", fig, step)
            plt.close(fig)

            if step > 0:
                scheduler.step()
                reset_jitter_()
