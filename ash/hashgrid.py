import ash
import open3d as o3d
import open3d.core as o3c
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from .common import _get_c_extension, DotDict

_C = _get_c_extension()


import open3d as o3d
import open3d.core as o3c
from torch.utils.dlpack import from_dlpack, to_dlpack


def from_o3d(tensor):
    return from_dlpack(tensor.to_dlpack())


def to_o3d(tensor):
    return o3c.Tensor.from_dlpack(to_dlpack(tensor))


class Query(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        feat_params,
        hashmap,
        block_stride,
        level_offset,
        level_resolution,
        level_voxel_size,
        compute_dfeats_dx,
    ):
        feats, dfeats_dx, masks = _C.query_forward(
            hashmap.impl,
            x,
            feat_params,
            block_stride,
            level_offset,
            level_resolution,
            level_voxel_size,
            compute_dfeats_dx,
        )
        torch.cuda.synchronize()

        # TODO: check if returning masks is necessary
        ctx.save_for_backward(x, feat_params, feats, dfeats_dx, masks)
        ctx.hashmap = hashmap
        ctx.params = (
            block_stride,
            level_offset,
            level_resolution,
            level_voxel_size,
            compute_dfeats_dx,
        )

        return feats, dfeats_dx, masks

    @staticmethod
    def backward(ctx, dy_dfeats, dy_ddfeats_dx, dy_dmasks):
        x, feat_params, feats, dfeats_dx, masks = ctx.saved_tensors
        (
            block_stride,
            level_offset,
            level_resolution,
            level_voxel_size,
            compute_dfeats_dx,
        ) = ctx.params

        # In-place modify output_grad_feats
        dy_dx, dy_dfeat_params = _C.query_backward(
            ctx.hashmap.impl,
            x,
            feat_params,
            dfeats_dx if compute_dfeats_dx else torch.empty(1),  # Placeholder
            masks,
            dy_dfeats,
            dy_ddfeats_dx if compute_dfeats_dx else torch.empty(1),  # Placeholder,
            block_stride,
            level_offset,
            level_resolution,
            level_voxel_size,
            compute_dfeats_dx,
        )
        torch.cuda.synchronize()

        # Need to empty the cache for cached indices and weights
        # torch.cuda.empty_cache()

        # dy_dx TBD
        return None, dy_dfeat_params, None, None, None, None, None, None


class HashGrid(nn.Module):
    """
    Two usages:
    1) initialization. Allocate a hashmap with its value buffer.
       Save the hash map heap, and the values as params
    2) refinement. Load a hashmap with its value buffer in this case. Optionally
    """

    def __init__(
        self,
        capacity=10,
        voxel_size=0.01,
        key_ref=torch.zeros(3, dtype=torch.int32),
        dict_value_ref={
            "feature": torch.zeros(1 + 1 + 3, dtype=torch.float32),
        },
        resolutions=[8, 2],
        device=torch.device("cuda:0"),
    ):

        super().__init__()

        # Params
        self.capacity = capacity
        self.voxel_size = voxel_size
        self.resolutions = resolutions

        self.dim = len(key_ref)
        self.voxel_sizes = [voxel_size * resolutions[0] / r for r in resolutions]
        self.num_voxels = [r**self.dim for r in resolutions]
        self.stride = np.sum(self.num_voxels)
        self.offsets = [0, *np.cumsum(self.num_voxels)[0:-1]]

        self.device = device

        # Key values
        self.key_ref = key_ref
        self.dict_value_ref = dict_value_ref

        self.block_dict_value_ref = {}
        for k, v in dict_value_ref.items():
            self.block_dict_value_ref[k] = torch.zeros(
                (self.stride, *v.shape), dtype=v.dtype
            )

        self.reset_hashmap(capacity, device)
        self.reset_parameters(dict_value_ref)

        # Hash map iterator
        self.block_coords = None
        self.block_indices = None

    def reset_hashmap(self, capacity, device):
        self.hashmap = ash.HashMap(
            capacity,
            key_ref=self.key_ref,
            dict_value_ref=self.block_dict_value_ref,
            device=device,
        )

    def reset_parameters(self, dict_value_ref=None):
        if dict_value_ref is None:
            dict_value_ref = self.dict_value_ref
        self.params = nn.ParameterDict({})
        for k in dict_value_ref:
            self.params[k] = nn.Parameter(self.hashmap.value(k))

    def update_items_(self):
        self.block_coords, self.block_indices = self.hashmap.items()

    def resize(self, new_capacity):
        self.hashmap.resize(new_capacity)

        self.capacity = self.hashmap.capacity
        for k in self.hashmap.names:
            self.params[k] = nn.Parameter(self.hashmap.value(k))

    # Query ops
    def query(self, input_coords, level, name="feature", compute_dfeats_dx=False):
        return Query.apply(
            input_coords,
            self.params[name],
            self.hashmap,
            self.stride,
            self.offsets[level],
            self.resolutions[level],
            self.voxel_sizes[level],
            compute_dfeats_dx,
        )

    def clamp_(self):
        with torch.no_grad():
            torch.clamp_(self.params["feature"][..., 0], -1, 1)
            torch.clamp_(self.params["feature"][..., 2:], 0, 1)

    def forward(self, input_coords, level=0, name="feature"):
        # (N, d), (N, d, 3), (N, 1)
        sample_feat, sample_feat_gradient, sample_mask = self.query(
            input_coords, level, name=name, compute_dfeats_dx=True
        )

        return DotDict(
            {
                "features": sample_feat,
                "gradients": sample_feat_gradient,
                "masks": sample_mask,
            }
        )

    def conv_forward(self, weights, start_dim, end_dim, level=0, name="feature"):
        # TODO: detailed dim check
        assert len(weights.shape) == self.dim + 1
        assert weights.shape[-1] == end_dim - start_dim

        C = weights.shape[0]
        assert C > 0 and C % 2 == 1
        for i in range(0, self.dim):
            assert weights.shape[i] == C

        voxel_nb_radius = C // 2
        block_nb_radius = (
            voxel_nb_radius + self.resolutions[level] - 1
        ) // self.resolutions[level]
        block_coords, block_indices = self.hashmap.items()

        out_feat_params = torch.zeros_like(self.params[name][..., start_dim:end_dim])

        # Store cache for neighbors
        # TODO: cache it if multiple times are needed
        block_nb_indices, block_nb_masks, _ = self.hashmap.radius_find_dense(
            block_coords, radius=block_nb_radius
        )

        _C.conv_forward_dense(
            self.params[name][..., start_dim:end_dim].contiguous(),
            weights,
            block_coords,
            block_indices,
            block_nb_indices,
            block_nb_masks,
            out_feat_params,
            block_nb_radius,
            voxel_nb_radius,
            self.stride,
            self.offsets[level],
            self.resolutions[level],
        )
        return out_feat_params

    # Activation, pruning, and enumeration ops
    def activate(self, xyz, radius=1, temp_hashmap_capacity_multiplier=1):
        block_size = self.resolutions[0] * self.voxel_sizes[0]

        hashset = ash.HashMap(
            len(xyz) * temp_hashmap_capacity_multiplier,
            key_ref=self.key_ref,
            dict_value_ref={"dummy": torch.zeros(1, dtype=torch.int8)},
            device=self.device,
        )
        keys = torch.floor(xyz / block_size).to(self.key_ref.dtype)

        if radius > 0:
            hashset.radius_activate(keys, radius)
        else:
            hashset.activate(keys)

        # Get unique coordinates
        block_coords, _ = hashset.items()
        self.hashmap.activate(block_coords)

        return block_coords

    def prune(
        self, weight_threshold=3.0, distance_threshold=0.5, percentage=0.8, level=0
    ):
        return _C.prune(
            self.hashmap.impl,
            self.hashmap.value("feature")[..., 0:1],
            self.hashmap.value("feature")[..., 1:2],
            self.stride,
            self.offsets[level],
            self.resolutions[level],
            float(weight_threshold),
            distance_threshold,
            percentage,
        )

    def voxel_items(self, block_coords, block_indices, level):
        input_positions, voxel_indices = _C.voxel_items(
            self.hashmap.impl,
            block_coords,
            block_indices,
            self.stride,
            self.offsets[level],
            self.resolutions[level],
        )
        return input_positions, voxel_indices

    @staticmethod
    def from_state_dict(state_dict, device=None):
        return HashGrid(
            capacity=state_dict["conf.capacity"],
            voxel_size=state_dict["conf.voxel_size"],
            key_ref=state_dict["conf.key_ref"],
            dict_value_ref=state_dict["conf.dict_value_ref"],
            resolutions=state_dict["conf.resolution"],
            device=torch.device(state_dict["conf.device"])
            if device is None
            else device,
        )

    def state_dict(self):
        state_dict = super().state_dict()

        # Additional properties other than params
        # DO NOT store values -- they are explicitly stored as params
        hashmap_state_dict = self.hashmap.state_dict(store_values=False)
        hashmap_state_dict["conf.dict_value_ref"] = self.dict_value_ref
        hashmap_state_dict["conf.voxel_size"] = self.voxel_size
        hashmap_state_dict["conf.resolution"] = self.resolutions

        for k, v in hashmap_state_dict.items():
            state_dict[k] = v

        return state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        device = self.device

        # Load key-index mapping
        self.hashmap.load_state_dict(state_dict, load_values=False)
        self.reset_parameters()

        # Resize parameters if necessary

        # Explicitly load values
        for k, v in state_dict.items():
            if k.startswith("params"):
                param_name = k.split(".")[-1]
                assert param_name in self.params

                # TODO: serialize if direct assignment is too memory consuming
                self.params[param_name][:] = v.to(device)

    def extract_pointcloud(self, isovalue=0.0, level=0, weight_threshold=1.0):
        positions = _C.extract_isosurface_positions(
            self.hashmap.impl,
            self.hashmap.value("feature")[..., 0:1].contiguous(),
            self.hashmap.value("feature")[..., 1:2].contiguous(),
            self.stride,
            self.offsets[level],
            self.resolutions[level],
            self.voxel_sizes[level],
            isovalue,
            float(weight_threshold),
        )

        with torch.no_grad():
            ans = self(positions, name="feature")
            normals = F.normalize(ans["gradients"][:, 0, :].contiguous(), dim=1)
            colors = ans["features"][..., 2:].contiguous()

        pcd = o3d.t.geometry.PointCloud(to_o3d(positions))
        pcd.point["colors"] = to_o3d(colors)
        pcd.point["normals"] = to_o3d(normals)
        return pcd

    def marching_cubes(self, isovalue=0.0, level=0, weight_threshold=1.0):
        # TODO: triangles and positions
        print(self.hashmap.value("feature").shape)
        positions, triangles = _C.marching_cubes(
            self.hashmap.impl,
            self.hashmap.value("feature")[..., 0:1].contiguous(),
            self.hashmap.value("feature")[..., 1:2].contiguous(),
            self.stride,
            self.offsets[level],
            self.resolutions[level],
            self.voxel_sizes[level],
            isovalue,
            float(weight_threshold),
        )

        with torch.no_grad():
            ans = self(positions, name="feature")
            normals = F.normalize(ans["gradients"][:, 0, :].contiguous(), dim=1)
            colors = ans["features"][..., 2:].contiguous()

        mesh = o3d.t.geometry.TriangleMesh()
        mesh.vertex["positions"] = to_o3d(positions)
        mesh.vertex["colors"] = to_o3d(colors)
        mesh.vertex["normals"] = to_o3d(normals)
        mesh.triangle["indices"] = to_o3d(triangles)
        return mesh.to_legacy()

    def to_lineset(self, color=[0, 0, 1], scale=0.1):
        xyz000, _ = self.hashmap.items()

        block_len = self.voxel_sizes[0] * self.resolutions[0]

        xyz000 = xyz000.cpu().numpy() * block_len
        xyz001 = xyz000 + np.array([[block_len * scale, 0, 0]])
        xyz010 = xyz000 + np.array([[0, block_len * scale, 0]])
        xyz100 = xyz000 + np.array([[0, 0, block_len * scale]])
        xyz = np.concatenate((xyz000, xyz001, xyz010, xyz100), axis=0).astype(
            np.float32
        )

        lineset = o3d.t.geometry.LineSet()
        lineset.point["positions"] = o3c.Tensor(xyz)

        n = len(xyz000)
        lineset000 = np.arange(0, n)
        lineset001 = np.arange(n, 2 * n)
        lineset010 = np.arange(2 * n, 3 * n)
        lineset100 = np.arange(3 * n, 4 * n)

        indices001 = np.stack((lineset000, lineset001), axis=1)
        indices010 = np.stack((lineset000, lineset010), axis=1)
        indices100 = np.stack((lineset000, lineset100), axis=1)
        indices = np.concatenate((indices001, indices010, indices100), axis=0)

        lineset.line["indices"] = o3c.Tensor(indices.astype(np.int32))
        colors = np.tile(color, (3 * n, 1))
        lineset.line["colors"] = o3c.Tensor(colors.astype(np.float32))
        return lineset

    # Legacy
    def to_o3d_voxelblockgrid(self, level):
        vbg = o3d.t.geometry.VoxelBlockGrid(
            ("tsdf", "weight", "color"),
            (o3c.float32, o3c.float32, o3c.float32),
            ((1), (1), (3)),
            self.voxel_sizes[level],
            self.resolutions[level],
            int(self.hashmap.size() * 1.2),
            o3c.Device(str(self.device)),
        )

        block_coords, block_indices = self.hashmap.items()
        block_coords = to_o3d(block_coords.int())

        start = self.offsets[level]
        end = None if level + 1 >= len(self.resolutions) else self.offsets[level + 1]

        tsdf = (
            self.hashmap.value("feature")[block_indices, start:end, 0]
            .view(len(block_coords), self.num_voxels[level])
            .contiguous()
        )
        tsdf = to_o3d(tsdf)

        weight = (
            self.hashmap.value("feature")[block_indices, start:end, 1]
            .view(len(block_coords), self.num_voxels[level])
            .contiguous()
        )
        weight = to_o3d(weight)

        color = (
            self.hashmap.value("feature")[block_indices, start:end, 2:]
            .view(len(block_coords), self.num_voxels[level], 3)
            .contiguous()
        )
        color = to_o3d(color) * 255.0

        vbg_hashmap = vbg.hashmap()
        vbg_hashmap.insert(block_coords, [tsdf, weight, color])

        return vbg
