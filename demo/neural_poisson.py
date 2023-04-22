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
import trimesh
from skimage import measure

avg_pool_3d = torch.nn.AvgPool3d(2, stride=2)
upsample = torch.nn.Upsample(scale_factor=2, mode="nearest")


def get_surface_sliding(
    sdf_fn, fname, resolution=100, grid_boundary=[-1.0, 1.0], level=0
):
    assert resolution % 512 == 0
    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = [grid_boundary[0], grid_boundary[0], grid_boundary[0]]
    grid_max = [grid_boundary[1], grid_boundary[1], grid_boundary[1]]
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = np.linspace(x_min, x_max, cropN)
                y = np.linspace(y_min, y_max, cropN)
                z = np.linspace(z_min, z_max, cropN)

                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(
                    np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float
                ).cuda()

                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 100000, dim=0)):
                        z.append(sdf_fn(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3).permute(3, 0, 1, 2)
                points_pyramid = [points]
                for _ in range(3):
                    points = avg_pool_3d(points[None])[0]
                    points_pyramid.append(points)
                points_pyramid = points_pyramid[::-1]

                # evalute pyramid with mask
                mask = None
                threshold = 2 * (x_max - x_min) / cropN * 8
                for pid, pts in enumerate(points_pyramid):
                    coarse_N = pts.shape[-1]
                    pts = pts.reshape(3, -1).permute(1, 0).contiguous()

                    if mask is None:
                        pts_sdf = evaluate(pts)
                    else:
                        mask = mask.reshape(-1)
                        pts_to_eval = pts[mask]
                        # import pdb; pdb.set_trace()
                        if pts_to_eval.shape[0] > 0:
                            pts_sdf_eval = evaluate(pts_to_eval.contiguous())
                            pts_sdf[mask] = pts_sdf_eval

                    if pid < 3:
                        # update mask
                        mask = torch.abs(pts_sdf) < threshold
                        mask = mask.reshape(coarse_N, coarse_N, coarse_N)[None, None]
                        mask = upsample(mask.float()).bool()

                        pts_sdf = pts_sdf.reshape(coarse_N, coarse_N, coarse_N)[
                            None, None
                        ]
                        pts_sdf = upsample(pts_sdf)
                        pts_sdf = pts_sdf.reshape(-1)

                    threshold /= 2.0

                z = pts_sdf.detach().cpu().numpy()

                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    try:
                        verts, faces, normals, values = measure.marching_cubes(
                            volume=z.reshape(
                                cropN, cropN, cropN
                            ),  # .transpose([1, 0, 2]),
                            level=level,
                            spacing=(
                                (x_max - x_min) / (cropN - 1),
                                (y_max - y_min) / (cropN - 1),
                                (z_max - z_min) / (cropN - 1),
                            ),
                        )
                        verts = verts + np.array([x_min, y_min, z_min])
                        meshcrop = trimesh.Trimesh(verts, faces, normals)
                        # meshcrop.export(f"{i}_{j}_{k}.ply")
                        meshes.append(meshcrop)

                    except:
                        continue

    combined = trimesh.util.concatenate(meshes)
    try:
        combined.export(fname, "ply")
    except:
        print("empty mesh")


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

        self.positions = (self.positions - self.center) * self.scale

        # For visualization
        self.pcd = o3d.t.geometry.PointCloud(self.positions)
        self.pcd.point.normals = self.normals
        # o3d.visualization.draw_geometries([self.pcd.to_legacy()])

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
            num_embeddings=12000,
            embedding_dim=embedding_dim,
            grid_dim=8,
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

    def marching_cubes(self, fname):
        def normal_fn(positions):
            positions.requires_grad_(True)
            density, grad_x, mask = self.forward(positions)
            return grad_x

        def sdf_fn():
            grid_coords, cell_coords, grid_indices, cell_indices = self.grid.items()
            cell_positions = self.grid.cell_to_world(grid_coords, cell_coords)
            cell_positions = cell_positions.view(grid_coords.shape[0], -1, 3)

            sdfs = torch.zeros(
                self.grid.embeddings.shape[0],
                self.grid.embeddings.shape[1],
                1,
                device=self.grid.embeddings.device,
            )
            weights = torch.zeros(
                self.grid.embeddings.shape[0],
                self.grid.embeddings.shape[1],
                1,
                device=self.grid.embeddings.device,
            )

            for i in tqdm(range(len(grid_coords))):
                positions = cell_positions[i]
                lhs = grid_indices[i]
                # print(sdfs.shape, sdfs[lhs].shape, self.forward(positions)[0].shape)
                sdf, grad_x, mask = self.forward(positions)
                sdfs[lhs] = sdf.view(-1, 512, 1).detach()
                weights[lhs] = mask.view(-1, 512, 1).detach().float()

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
        return mesh


class NeuralPoissonMLP(NeuralPoisson):
    def __init__(self, grid_resolution, device=torch.device("cuda:0")):
        super().__init__(
            grid_resolution=grid_resolution, embedding_dim=8, device=device
        )
        torch.nn.init.normal_(self.grid.embeddings)

        lin0 = torch.nn.Linear(8, 32)
        # torch.nn.init.constant_(lin0.bias, 0.0)
        # torch.nn.init.constant_(lin0.weight[:, 3:], 0.0)
        # torch.nn.init.normal_(lin0.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(32))

        lin2 = torch.nn.Linear(32, 32)
        # torch.nn.init.normal_(
        #     lin2.weight, mean=-np.sqrt(np.pi) / np.sqrt(32), std=0.0001
        # )
        # torch.nn.init.constant_(lin2.bias, 1.0)

        lin3 = torch.nn.Linear(32, 1)
        # torch.nn.init.normal_(lin3.weight, 0.0, np.sqrt(2) / np.sqrt(1))

        self.mlp = torch.nn.Sequential(
            lin0, torch.nn.ReLU(), lin2, torch.nn.ReLU(), lin3
        ).to(device)

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

    def marching_cubes(self, fname):
        def normal_fn(positions):
            positions.requires_grad_(True)
            grad_xs = []
            for i in range(0, len(positions), 10000):
                sdf, grad_x, mask = self.forward(positions[i : i + 10000])
                grad_xs.append(grad_x.detach())
            grad_x = torch.cat(grad_xs, dim=0)
            return grad_x.contiguous()

        def sdf_fn(positions):
            positions.requires_grad_(True)
            sdf, grad_x, mask = self.forward(positions)
            sdf = sdf.detach()
            sdf[~mask] = -1.0
            return sdf.squeeze()

        get_surface_sliding(sdf_fn, fname, resolution=512)

        # mesh = self.grid.marching_cubes(
        #     tsdfs=sdfs,
        #     weights=weights,
        #     color_fn=None,
        #     normal_fn=normal_fn,
        #     iso_value=0.0,
        #     vertices_only=False,
        #     weight_thr=1.0,
        # )
        # return mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    args = parser.parse_args()

    dataset = PointCloudDataset(args.path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10000, shuffle=True)

    model = NeuralPoissonPlain(grid_resolution=64)
    model.spatial_init_(torch.from_numpy(dataset.positions).cuda())
    print(model.grid.engine.size())

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model.marching_cubes("mesh_init.ply")
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

            loss_surface = sdf[mask].pow(2).mean()
            loss_surface_normal = (grad_x - normals)[mask].pow(2).sum(dim=-1).mean() + (
                (grad_x * normals)[mask].sum(dim=1) - 1
            ).pow(2).mean()
            loss_surface_eikonal = (torch.norm(grad_x[mask], dim=-1) - 1).pow(2).mean()

            rand_positions = model.sample_grid(num_samples=int(len(positions)))
            sdf_rand, grad_x_rand, mask_rand = model(rand_positions)

            loss_rand_sdf = (
                (model.grid.cell_size - sdf_rand[mask_rand].abs()).pow(2).mean()
            )
            loss_rand_eikonal = (
                (torch.norm(grad_x_rand[mask_rand], dim=-1) - 1).pow(2).mean()
            )

            normals = torch.ones_like(grad_x_rand) * -1
            loss_rand_normal = (grad_x_rand - normals)[mask_rand].pow(2).sum(
                dim=-1
            ).mean() + ((grad_x_rand * normals)[mask_rand].sum(dim=1) - 1).pow(2).mean()

            loss = (
                # loss_surface
                +loss_surface_normal
                # + loss_surface_eikonal
                # + loss_rand_sdf
                # + loss_rand_eikonal
            )
            loss.backward()

            pbar.set_description(
                f"Total={loss.item():.4f}, surface={loss_surface.item():.4f}, normal={loss_surface_normal.item():.4f}, eikonal={loss_surface_eikonal.item():.4f}, rand_sdf={loss_rand_sdf.item():.4f}, rand_eikonal={loss_rand_eikonal.item():.4f}"
            )
            optimizer.step()

        scheduler.step()
        model.marching_cubes(f"mesh_{epoch:03d}.ply")
        # if mesh is not None:
        #     o3d.io.write_triangle_mesh(f"mesh_{epoch:03d}.ply", mesh.to_legacy())
