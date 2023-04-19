# torch-ash
torch-ash is the missing piece of collision-free extendable parallel spatial hashing for torch modules. It includes two paper's core implementations:

```
@article{dong2022ash,
  title={ASH: A modern framework for parallel spatial hashing in 3D perception},
  author={Dong, Wei and Lao, Yixing and Kaess, Michael and Koltun, Vladlen},
  journal={PAMI},
  year={2022},
}

@inproceedings{dong2023ash-mono,
  title={Fast Monocular Scene Reconstruction with Global-Sparse Local-Dense Grids},
  author={Dong, Wei and Choy, Chris and Loop, Charles and Zhu, Yuke and Litany, Or and Anandkumar, Anima},
  booktitle={CVPR},
  year={2023},
}
```

## Install
cmake is required in the conda environment.
```
pip install . --verbose
```

## Basics
- The core is ASHEngine, which implements a hash map from coordinates (`torch.IntTensor`) to indices (`torch.LongTensor`).
- Above ASHEngine, there are `HashSet` and `HashMap` which are wrappers around ASHEngine. A `HashSet` maps a coordinate to a boolean value, usually used for the `unique` operation. A `HashMap` maps a coordinate to a (dictionary) of values, allows fast insertion and accessing coordinate-value pairs.
- Similar to `HashMap`, `HashEmbedding` maps coordinates to embeddings that is akin to `torch.nn.Embedding`.

## SparseDenseGrids
`SparseDenseGrid` is the engine for direct/neural reconstruction. It consists of sparse arrays of grids and dense arrays of cells. The idea is similar to instant-ngp, but true sparsity is achieved through spatial initialization and the collision-free hashing. 

It has two wrappers for coordinate transform, `UnboundedSparseDenseGrid` for potentially dynamically increasing metric scenes, and `BoundedSparseDenseGrid` for scenes bounded in unit cubes. Trilinear interpolation and double backwards are implemented to support differentiable gradient computation.

The general pattern for using the grid is:
```python
grid = UboundedSparseDenseGrid(in_dim=3, 
                               num_embeddings=10000,
                               grid_dim=16, 
                               embedding_dims=8, 
                               cell_size=0.01)

# Initialize grids by insertion 
for points in list_points:
    with torch.no_grad():
        grid_coords, cell_coords, grid_indices, cell_indices = grid.spatial_init_(points)
        
        # [Optional] direct assignment
        cell_points = grid.cell_to_world(grid_coords, cell_coords)
        estimate = estimate_fn(cell_points)
        grid.embeddings[grid_indices, cell_indices] = estimate


# Optimize via auto-differentiation
optim = torch.optim.SGD(grid.parameters(), lr=1e-3)
for x, gt in batch:
    optim.zero_grad()
    x.requires_grad_(True)
    embedding, mask = grid(x, interpolation="linear")

    output = geometry_mlp(embedding, mask)
    sdf, geo_features = torch.split(output, 1)
    rgb = color_mlp(positional_encoding(x), geo_features)

    dsdf_dposition = torch.autograd.grad(
        outputs=sdf, 
        inputs=x,
        grad_outputs=torch.ones_like(sdf, requires_grad=False),
        create_graph=True,
        retain_graph=True)[0]

    eikonal_loss = ((torch.norm(dsdf_dposition, dim=-1) - 1)**2).mean()
    data_loss = loss_fn(sdf, rgb, gt)
    (eikonal_loss + data_loss).backward()
    optim.step()
```

All these modules can be converted to and from state dicts by serializing the underlying hash map.

## MultiResSparseDenseGrids
TBD: a naive implementation would be simply maintaining `L` `SparseDenseGrid`s. If it has severe efficiency issues, optimize kernels accordingly.

## Experiments
- `SparseDenseGrids` with point clouds.
- `SparseDenseGrids` with RGB + monocular depth.
- `SparseDenseGrids` with RGB + monocular depth + point clouds.
- Elevate to `MultiResSparseDenseGrids` if results are problematic.
