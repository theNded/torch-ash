import open3d as o3d
from tqdm import tqdm

import numpy as np
import torch

from data_provider import ImageDataset, Dataloader, to_o3d

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--depth_type", default="sensor", choices=["sensor", "learned"])
    args = parser.parse_args()


    dataset = ImageDataset(
        args.path, depth_type=args.depth_type, normalize_scene=True
    )

    batch_size = dataset.H * dataset.W
    dataloader = Dataloader(dataset, batch_size=batch_size, shuffle=True)

    geometries = []
    for i in tqdm(range(20)):
        data = next(iter(dataloader))

        positions = data["rays_o"] + data["rays_d"] * (
            data["depth"] * data["depth_scale"] * data["rays_d_norm"]
        )
        rgbs = data["rgb"]
        normals = data["normal"]

        pcd = o3d.t.geometry.PointCloud(to_o3d(positions))
        pcd.point.colors = to_o3d(rgbs)
        pcd.point.normals = to_o3d(normals)

        geometries.append(pcd)

    o3d.visualization.draw(geometries)
