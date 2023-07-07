
import torch

from .sparsedense_grid import SparseDenseGrid


def marching_cubes(
    self, resolution, sdf_fn, weight_fn, bbox_min, bbox_max, grid_dim=8
):
    # TODO(wei): still pseudo-code ish, complete it soon
    # First create a coarse sparse-dense grid
    # Generate a meshgrid

    # Coarse resolution: 256
    sparse_resolution = resolution // grid_dim
    x = torch.linspace(bbox_min[0], bbox_max[0], sparse_resolution)
    y = torch.linspace(bbox_min[1], bbox_max[1], sparse_resolution)
    z = torch.linspace(bbox_min[2], bbox_max[2], sparse_resolution)
    xx, yy, zz = torch.meshgrid(x, y, z)

    sparse_keys = torch.stack((xx, yy, zz)).reshape(3, -1).T

    # Maybe use some batch tricks here to serialize
    sdfs = sdf_fn(sparse_keys)
    weights = weight_fn(sparse_keys)

    # Rough mask
    mask = (sdfs.abs() < 0.1) * (weights > 0.5)

    # Create a sparse-dense grid and initialize
    sparse_dense_grid = SparseDenseGrid(
        in_dim=3,
        num_embeddings=1,
        sparse_grid_dim=sparse_resolution,
        grid_dim=grid_dim,
    )
    sparse_dense_grid.spatial_init_(sparse_keys[mask])

    (
        grid_coords,
        cell_coords,
        grid_indices,
        cell_indices,
    ) = sparse_dense_grid.items()

    # Serialize query and assignment
    coords = grid_coords * grid_dim + cell_coords
    indices = grid_indices * grid_dim + cell_indices
    sdfs = sdf_fn(coords)
    weights = weight_fn(coords)
    sparse_dense_grid.embeddings[grid_indices, cell_indices] = torch.stack(
        (sdfs, weights)
    ).T

    # Now extract mesh
    # Here dimensions must be 1 (sdf only), 2 (sdf + weights), or 5 (sdf + weights + colors)
    faces, vertices, normals, colors = sparse_dense_grid.marching_cubes()

    # Convert to o3d mesh and draw
    mesh = o3d.t.geometry.TriangleMesh()
    mesh.vertex["positions"] = to_o3d(positions)
    mesh.vertex["colors"] = to_o3d(colors)
    mesh.vertex["normals"] = to_o3d(normals)
    mesh.triangle["indices"] = to_o3d(triangles)
    return mesh
