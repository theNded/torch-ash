import numpy as np

import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

import open3d as o3d
import open3d.core as o3c

import cv2

from pathlib import Path
from tqdm import tqdm


def to_o3d(tensor):
    return o3c.Tensor.from_dlpack(to_dlpack(tensor))


def from_o3d(tensor):
    return from_dlpack(tensor.to_dlpack())


def get_image_files(
    path, folders=["image", "color"], exts=["jpg", "png", "pgm", "npy"]
):
    for folder in folders:
        for ext in exts:
            image_fnames = sorted((path / folder).glob(f"*.{ext}"))
            if len(image_fnames) > 0:
                return image_fnames
    raise ValueError(f"no images found in {path}")


def load_image(fname, im_type="image"):
    """
    Load image from file.
    """
    if fname.suffix == ".npy":
        # Versatile, could work for any image type
        data = np.load(fname)
        if data.shape[0] in [1, 3]:  # normal or depth transposed
            data = data.transpose((1, 2, 0))
        if len(data.shape) == 2:  # depth squeezed
            data = np.expand_dims(data, axis=-1)
        return data
    elif fname.suffix in [".jpg", ".jpeg", ".png", ".pgm"]:
        if im_type == "image":
            # Always normalize RGB to [0, 1]
            return cv2.imread(str(fname), cv2.IMREAD_COLOR)[..., ::-1] / 255.0
        elif im_type == "depth":
            # Keep depth as they are as unit is usually tricky
            return cv2.imread(str(fname), cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f"unknown image type {im_type} with {fname.suffix}")
    else:
        raise ValueError(f"unknown image type {fname.suffix}")


class PointCloudDataset(torch.utils.data.Dataset):
    """Minimal point cloud dataset for a single point cloud
    Point clouds include:
        - positions: (N, 3) float32
        - normals: (N, 3) float32
    """

    def __init__(self, path, normalize_scene=True):
        self.path = Path(path)

        self.pcd = o3d.t.io.read_point_cloud(str(self.path))

        self.positions = self.pcd.point.positions.numpy()
        assert "normals" in self.pcd.point
        self.normals = self.pcd.point.normals.numpy()
        self.normals /= np.linalg.norm(self.normals, axis=1, keepdims=True)

        assert len(self.positions) == len(self.normals)
        self.num_points = len(self.positions)

        min_vertices = np.min(self.positions, axis=0)
        max_vertices = np.max(self.positions, axis=0)

        self.center = (min_vertices + max_vertices) / 2.0
        self.scale = 2.0 / (np.max(max_vertices - min_vertices) * 1.1)

        # Normalize the point cloud into [-1, 1] box
        if normalize_scene:
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


class ImageDataset:
    """Minimal RGBD dataset with known poses
    As a raw data container, the images can be accessed at [i] by image-wise
    - depth: (N, H, W) float32
    - color: (N, H, W, 3) float32
    (optional)
    - normal: (N, H, W) bool
    - semantic: (N, H, W, D) float32
    All in the camera coordinate systems.

    As a dataloader for ray-based rendering, the pixels can be loaded by dataloaders at [i] through pixel-wise
    - ray_o: (N * H * W, 3)
    - ray_d: (N * H * W, 3)
    - depth: (N * H * W, 1)
    - color: (N * H * W, 3)
    (optional)
    - normal: (N * H * W, 3)
    - semantic: (N * H * W, D)
    All transformed into the global coordinate systems since shuffle may result in further complexity.

    All the data are loaded to numpy in CPU memory.
    TODO: follow nerfstudio and add more flexibility with cached dataloader
    """

    # pixel value * depth_scale = depth in meters
    depth_scale = 1e-3
    depth_max = 4.0

    def __init__(self, path, normalize_scene=True):
        self.path = Path(path)

        # Load fnames
        self.image_fnames = get_image_files(
            self.path, folders=["image", "color"], exts=["png", "jpg"]
        )
        self.depth_fnames = get_image_files(
            self.path, folders=["depth"], exts=["png", "pgm", "npy"]
        )
        self.normal_fnames = get_image_files(
            self.path, folders=["omni_normal"], exts=["npy"]
        )

        # Load intrinsics and poses
        self.intrinsic = np.loadtxt(self.path / "intrinsic_depth.txt").reshape((3, 3))
        self.poses_unnormalized = np.loadtxt(self.path / "poses.txt").reshape(
            (-1, 4, 4)
        )

        assert len(self.image_fnames) == len(
            self.depth_fnames
        ), f"{len(self.image_fnames)} != {len(self.depth_fnames)}"
        assert len(self.image_fnames) == len(
            self.poses_unnormalized
        ), f"{len(self.image_fnames)} != {len(self.poses_unnormalized)}"
        if len(self.normal_fnames) > 0:
            assert len(self.image_fnames) == len(
                self.normal_fnames
            ), f"{len(self.image_fnames)} != {len(self.normal_fnames)}"

        # Load images and shapes
        depth_ims = []
        rgb_ims = []
        normal_ims = []
        pbar = tqdm(range(len(self.image_fnames)))
        for i in pbar:
            pbar.set_description(f"Loading {self.image_fnames[i]}")
            depth_ims.append(load_image(self.depth_fnames[i], "depth"))
            pbar.set_description(f"Loading {self.depth_fnames[i]}")
            rgb_ims.append(load_image(self.image_fnames[i], "image"))
            if len(self.normal_fnames) > 0:
                pbar.set_description(f"Loading {self.normal_fnames[i]}")
                normal_ims.append(load_image(self.normal_fnames[i], "omni_normal"))
            pbar.update()
        self.depth_ims = np.stack(depth_ims)
        self.rgb_ims = np.stack(rgb_ims)
        self.normal_ims = np.stack(normal_ims) if len(normal_ims) > 0 else None
        self.num_images, self.H, self.W = self.depth_ims.shape
        assert self.rgb_ims.shape == (self.num_images, self.H, self.W, 3)
        assert self.normal_ims is None or self.normal_ims.shape == (
            self.num_images,
            self.H,
            self.W,
            3,
        )

        # Normalize the scene if necessary
        self.bbox_T_world = np.eye(4)
        self.center = np.zeros(3, dtype=np.float32)
        self.scale = 1.0

        if normalize_scene:
            min_vertices = self.poses_unnormalized[:, :3, 3].min(axis=0)
            max_vertices = self.poses_unnormalized[:, :3, 3].max(axis=0)
            self.center = (min_vertices + max_vertices) / 2.0

            self.scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
            self.depth_scale *= self.scale
            self.depth_max *= self.scale
            print(f"Normalize scene with scale {self.scale} and center {self.center}m")

        poses = []
        extrinsics = []
        for pose in self.poses_unnormalized:
            pose[:3, 3] = (pose[:3, 3] - self.center) * self.scale
            poses.append(pose)
            extrinsics.append(np.linalg.inv(pose))
        self.poses = np.stack(poses)
        self.extrinsics = np.stack(extrinsics)

        # Generate rays
        yy, xx = np.meshgrid(np.arange(self.H), np.arange(self.W), indexing="ij")
        yy = yy.flatten()
        xx = xx.flatten()

        # (H*W, 3)
        rays = np.stack((xx, yy, np.ones_like(xx)), axis=-1).reshape(-1, 3)

        self.rays_d = []
        self.rays_d_norm = []
        self.rays_o = []
        pbar = tqdm(range(len(self.image_fnames)))
        for i in pbar:
            pbar.set_description(f"Generating rays for image {i}")
            P = self.intrinsic @ self.extrinsics[i, :3, :3]
            inv_P = np.linalg.inv(P)

            # Note: this rays_d is un-normalized
            rays_d = np.matmul(rays, inv_P.T)
            rays_d_norm = np.linalg.norm(rays_d, axis=-1, keepdims=True)
            rays_o = np.tile(self.poses[i, :3, 3], (self.H * self.W, 1))

            self.rays_o.append(rays_o)
            self.rays_d.append(rays_d / rays_d_norm)
            self.rays_d_norm.append(rays_d_norm)

        self.rays_d = (
            np.concatenate(self.rays_d, axis=0).reshape(-1, 3).astype(np.float32)
        )
        self.rays_d_norm = (
            np.concatenate(self.rays_d_norm, axis=0).reshape(-1, 1).astype(np.float32)
        )
        self.rays_o = (
            np.concatenate(self.rays_o, axis=0).reshape(-1, 3).astype(np.float32)
        )

        self.depths = self.depth_ims.reshape(-1, 1).astype(np.float32)
        self.rgbs = self.rgb_ims.reshape(-1, 3).astype(np.float32)
        if self.normal_ims is not None:
            self.normals = []
            pbar = tqdm(range(len(self.image_fnames)))
            for i in pbar:
                pbar.set_description(f"Transforming normals for image {i}")
                normals = self.normal_ims[i].reshape(-1, 3) @ self.poses[i, :3, :3].T
                self.normals.append(normals)
            self.normals = (
                np.concatenate(self.normals, axis=0).reshape(-1, 3).astype(np.float32)
            )

    def __len__(self):
        return len(self.rays_d)

    def __getitem__(self, idx):
        view_idx = idx // (self.H * self.W)
        return {
            "view_idx": view_idx,
            "rays_o": self.rays_o[idx],  # (N, 3)
            "rays_d": self.rays_d[idx],  # (N, 3)
            "rays_d_norm": self.rays_d_norm[idx],  # (N, 1)
            "depth": self.depths[idx],  # (N, 1)
            "depth_scale": self.depth_scale,
            "rgb": self.rgbs[idx],  # (N, 3)
            "normal": self.normals[idx]
            if self.normal_ims is not None
            else None,  # (N, 3)
        }


class Dataloader:
    def __init__(self, dataset, batch_size, shuffle, device=torch.device("cpu")):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self._generate_indices()

    def _generate_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self.dataset))
        else:
            indices = np.arange(len(self.dataset))

        self.num_batches = (
            len(self.dataset) + self.batch_size - 1
        ) // self.batch_size  # Round up to the nearest integer

        self.batch_indices = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, len(self.dataset))
            self.batch_indices.append(indices[start:end])

        self.batch_idx = 0

    def __getitem__(self, indices):
        batch = self.dataset.__getitem__(indices)
        for k, v in batch.items():
            # TODO: use key check instead
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v).to(self.device)
            elif isinstance(v, float):  # depth_scale
                batch[k] = torch.tensor(v).unsqueeze(-1).float().to(self.device)
            elif isinstance(v, int):  # view_idx
                batch[k] = torch.tensor(v).to(self.device)
        return batch

    def __iter__(self):
        indices = self.batch_indices[self.batch_idx]

        self.batch_idx += 1
        if self.batch_idx == self.num_batches:
            self.batch_idx = 0
            if self.shuffle:
                self._generate_indices()

        yield self.__getitem__(indices)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["pointcloud", "image"])
    parser.add_argument("--path", required=True)
    args = parser.parse_args()

    if args.dataset == "image":
        dataset = ImageDataset(args.path, normalize_scene=True)

        batch_size = dataset.H * dataset.W
        dataloader = Dataloader(dataset, batch_size=batch_size, shuffle=True)

        geometries = []
        for i in tqdm(range(20)):
            data = next(iter(dataloader))

            positions = data["rays_o"] + data["rays_d"] * (
                data["depth"] * data["rays_d_norm"] * data["depth_scale"]
            )
            rgbs = data["rgb"]
            normals = data["normal"]

            print(positions.dtype, rgbs.dtype, normals.dtype)
            pcd = o3d.t.geometry.PointCloud(to_o3d(positions))
            pcd.point.colors = to_o3d(rgbs)
            pcd.point.normals = to_o3d(normals)

            geometries.append(pcd)

        o3d.visualization.draw(geometries)

    elif args.dataset == "pointcloud":
        dataset = PointCloudDataset(args.path, normalize_scene=True)

        batch_size = dataset.num_points // 50
        dataloader = Dataloader(dataset, batch_size=batch_size, shuffle=True)

        geometries = []
        for i in tqdm(range(20)):
            data = next(iter(dataloader))

            positions = data["position"]
            normals = data["normal"]

            pcd = o3d.t.geometry.PointCloud(to_o3d(positions))
            pcd.point.normals = to_o3d(normals)

            geometries.append(pcd)
        o3d.visualization.draw(geometries)
