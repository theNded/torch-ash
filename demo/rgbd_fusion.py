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


def to_o3d(tensor):
    return o3c.Tensor.from_dlpack(to_dlpack(tensor))


def from_o3d(tensor):
    return from_dlpack(tensor.to_dlpack())


def to_o3d_im(tensor):
    return o3d.t.geometry.Image(to_o3d(tensor))


def get_image_files(path, folders=["image", "color"], exts=["jpg", "png", "pgm"]):
    for folder in folders:
        for ext in exts:
            image_fnames = sorted((path / folder).glob(f"*.{ext}"))
            if len(image_fnames) > 0:
                return image_fnames
    raise ValueError(f"no images found in {path}")


class RGBDDataset(torch.utils.data.Dataset):
    """Minimal RGBD dataset for testing purposes"""

    # pixel value / depth_scale = depth in meters
    depth_scale = 1000.0
    depth_max = 3.0

    def __init__(self, path, normalize_scene=True):
        self.path = Path(path)

        self.image_fnames = get_image_files(self.path, folders=["image", "color"])
        self.depth_fnames = get_image_files(self.path, folders=["depth"])
        self.poses = (
            np.loadtxt(self.path / "poses.txt").reshape((-1, 4, 4)).astype(np.float32)
        )
        self.intrinsic = (
            np.loadtxt(self.path / "intrinsic_depth.txt")
            .reshape((3, 3))
            .astype(np.float32)
        )

        assert len(self.image_fnames) == len(
            self.depth_fnames
        ), f"{len(self.image_fnames)} != {len(self.depth_fnames)}"
        assert len(self.image_fnames) == len(
            self.poses
        ), f"{len(self.image_fnames)} != {len(self.poses)}"

        self.bbox_T_world = np.eye(4)
        min_vertices = self.poses[:, :3, 3].min(axis=0)
        max_vertices = self.poses[:, :3, 3].max(axis=0)

        self.center = (min_vertices + max_vertices) / 2.0
        self.scale = 1.0

        if normalize_scene:
            self.scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
            self.depth_scale /= self.scale
            self.depth_max *= self.scale

            print(f"Normalize scene with scale {self.scale} and center {self.center}m")

        extrinsics = []
        for pose in self.poses:
            pose[:3, 3] = (pose[:3, 3] - self.center) * self.scale
            extrinsics.append(np.linalg.inv(pose))
        self.extrinsics = np.stack(extrinsics)

        # Debug
        bbox = o3d.geometry.AxisAlignedBoundingBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        bbox_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)

        camera_linesets = []
        for extrinsic in self.extrinsics:
            camera_linesets.append(
                o3d.geometry.LineSet.create_camera_visualization(
                    640, 480, self.intrinsic[:3, :3], extrinsic, 0.1
                )
            )
        o3d.visualization.draw([bbox_lineset] + camera_linesets)

    def __len__(self):
        return len(self.image_fnames)

    def __getitem__(self, idx):
        return {
            "color": cv2.imread(str(self.image_fnames[idx]))[..., ::-1].astype(
                np.float32
            )
            / 255.0,  # BGR2RGB
            "depth": cv2.imread(
                str(self.depth_fnames[idx]), cv2.IMREAD_ANYDEPTH
            ).astype(np.float32)
            / self.depth_scale,
            "intrinsic": self.intrinsic.astype(np.float32),
            "extrinsic": self.extrinsics[idx].astype(np.float32),
            "depth_scale": self.depth_scale,
            "depth_max": self.depth_max,
        }


class TSDFFusion:
    """Use ASH's hashgrid to generate differentiable sparse-dense grid from RGB + scaled monocular depth prior"""

    device = torch.device("cuda:0")

    def __init__(
        self,
        voxel_size: float = 0.01,
        normalize_scene: bool = True,
    ):
        if not normalize_scene:
            self.grid = UnBoundedSparseDenseGrid(
                in_dim=3,
                num_embeddings=100000,
                embedding_dim=5,
                grid_dim=8,
                cell_size=voxel_size,
                device=self.device,
            )
            self.voxel_size = voxel_size
            print(f"Use original scale scene with metric voxel size {self.voxel_size}m")

        else:
            self.grid = BoundedSparseDenseGrid(
                in_dim=3,
                num_embeddings=100000,
                embedding_dim=5,
                grid_dim=8,
                sparse_grid_dim=64,
                bbox_min=-1 * torch.ones(3, device=self.device),
                bbox_max=torch.ones(3, device=self.device),
                device=self.device,
            )

            self.voxel_size = self.grid.cell_size[0]
            print(
                f"Use normalized scene with non-metric voxel size {self.voxel_size} in bounding box"
            )

        self.trunc = 4 * self.voxel_size

    @torch.no_grad()
    def fuse_dataset(self, dataset):
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, shuffle=False
        )

        pbar = tqdm(total=len(dataloader))
        for i, datum in enumerate(dataloader):
            pbar.update(1)
            for k, v in datum.items():
                if isinstance(v, torch.Tensor):
                    datum[k] = v.to(self.device)
            datum = DotDict(datum)
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

        bbox = o3d.geometry.AxisAlignedBoundingBox([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        bbox_lineset = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
        # o3d.visualization.draw(
        #     [
        #         pcd,
        #         bbox_lineset,
        #         o3d.geometry.LineSet.create_camera_visualization(
        #             640,
        #             480,
        #             intrinsic[:3, :3].cpu().numpy(),
        #             extrinsic.cpu().numpy(),
        #             0.1,
        #         ),
        #     ]
        # )
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
    def prune(self, weight_threshold=1):
        pass

    @torch.no_grad()
    def fuse_frame(self, datum):
        torch.cuda.empty_cache()

        points = self.unproject_depth_to_points(
            datum.depth,
            datum.intrinsic,
            datum.extrinsic,
            datum.depth_scale,
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
        ) = self.grid.spatial_init_(points)
        cell_coords = self.grid.cell_to_world(grid_coords, cell_coords)

        # Observation
        sdf, rgb, w = self.project_points_to_rgbd(
            cell_coords,
            datum.intrinsic,
            datum.extrinsic,
            datum.color,
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

    def marching_cubes(self):
        triangles, positions = self.grid.marching_cubes()
        mesh.vertex["positions"] = to_o3d(positions)
        mesh.triangle["indices"] = to_o3d(triangles)
        return mesh.to_legacy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--normalize_scene",
        action="store_true",
        help="Normalize scene into the [0, 1] bounding box",
    )
    args = parser.parse_args()

    # Load data
    dataset = RGBDDataset(args.path, args.normalize_scene)
    fuser = TSDFFusion(0.01, args.normalize_scene)

    fuser.fuse_dataset(dataset)

    # sdf_fn and weight_fn
    sdf = fuser.grid.embeddings[..., 0].contiguous()
    weight = fuser.grid.embeddings[..., -1].contiguous()

    positions = fuser.grid.marching_cubes(sdf, weight, vertices_only=True)

    embeddings, masks = fuser.grid(positions, interpolation="linear")
    colors = embeddings[..., 1:4]
    pcd = o3d.t.geometry.PointCloud(positions.cpu().numpy())
    pcd.point["colors"] = colors.detach().cpu().numpy()
    o3d.visualization.draw(pcd)

    triangles, positions = fuser.grid.marching_cubes(sdf, weight, vertices_only=False)

    grid = fuser.grid
    optim = torch.optim.Adam(grid.parameters(), lr=1e-4)

    pbar = tqdm(range(100))
    for i in pbar:
        optim.zero_grad()

        positions.requires_grad_(True)
        embeddings, masks = grid(positions, interpolation="linear")

        dsdf_dx = torch.autograd.grad(
            outputs=embeddings[..., 0],
            inputs=positions,
            grad_outputs=torch.ones_like(embeddings[..., 0], requires_grad=False),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        eikonal_loss = ((torch.norm(dsdf_dx, dim=-1) - 1) ** 2).mean()
        pbar.set_description(f"iteration: {i}, loss: {eikonal_loss.item():.4f}")

        eikonal_loss.backward()

        optim.step()

    colors = embeddings[..., 1:4]

    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex["positions"] = positions.detach().cpu().numpy()
    mesh.vertex["colors"] = colors.detach().cpu().numpy()
    mesh.vertex["normals"] = F.normalize(dsdf_dx, dim=-1).detach().cpu().numpy()
    mesh.triangle["indices"] = triangles.cpu().numpy()
    o3d.visualization.draw([mesh.to_legacy()])
