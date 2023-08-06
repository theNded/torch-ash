# torch-ash
torch-ash is the missing piece of collision-free extendable parallel spatial hashing for torch modules. It includes two paper's core implementations:

[[PAMI 2022]](https://arxiv.org/abs/2110.00511) | [[CVPR 2023]](https://arxiv.org/abs/2305.13220)
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
<table>
  <tr>
  <td><img src="https://github.com/theNded/theNded.github.io/blob/master/assets/images/ash-mono/0050.gif" width="480"/></td>
  <td><img src="https://github.com/theNded/theNded.github.io/blob/master/assets/images/ash-mono/lounge.gif" width="480"/></td>
  </tr>
</table>


Note for a more user-friendly interface and further extensions, I have fully rewritten everything from scratch in this repo. Discrepancies from the reported results in the aforementioned papers are expected. Updates and more examples will come. 


## Install
First, install [PyTorch](https://pytorch.org/get-started/locally/).
Optionally install [nerfacc](https://pypi.org/project/nerfacc/0.2.1/) for volume rendering.

cmake is required in the conda environment for compiling the source code.
```
pip install . --verbose
```

## Engine (ASH)
- The core is `ASHEngine`, a PyTorch module implementing a parallel, collision-free, dynamic hash map from coordinates (`torch.IntTensor`) to indices (`torch.LongTensor`). It depends on [stdgpu](https://github.com/stotko/stdgpu).
- Above `ASHEngine`, there are `HashSet` and `HashMap` which are wrappers around ASHEngine. A `HashSet` maps a coordinate to a boolean value, usually used for the `unique` operation. A `HashMap` maps a coordinate to a (dictionary) of values, and allows fast insertion and accessing coordinate-value pairs.
- Similar to `HashMap`, `HashEmbedding` maps coordinates to embeddings and is akin to `torch.nn.Embedding`.

### Usage
```python
hashmap = HashMap(key_dim=3, value_dims={"color": 3, "depth": 1}, capacity=100, device=torch.device("cuda:0"))

# To insert
keys = (torch.rand(10, 3) * 100).int().cuda()
values = {"colors": torch.rand(10, 3).float().cuda(), "depth": torch.rand(10, 1).float().cuda()}
hashmap.insert(keys, values)

# To query
query_keys = (torch.rand(10, 3) * 100).int().cuda()
indices, masks = hashmap.find(query_keys)

# To enumerate
all_indices, all_values = hashmap.items(return_indices=True, return_values=True)
```

## SparseDenseGrids for Surface Reconstruction
`SparseDenseGrid` is the engine for direct/neural scene representation. It consists of sparse arrays of grids and dense arrays of cells. The idea is similar to [Instant-NGP](https://github.com/NVlabs/instant-ngp) and [Plenoxels](https://github.com/sxyu/svox2), but precise sparsity is achieved through spatial initialization and collision-free hashing. Essentially it is a modern version of [VoxelHashing](https://github.com/niessner/VoxelHashing).

It has two wrappers for coordinate transform, `UnboundedSparseDenseGrid` for potentially dynamically increasing metric scenes, and `BoundedSparseDenseGrid` for scenes bounded in unit cubes. Trilinear interpolation and double backward are implemented to support differentiable gradient computation. All these modules can be converted to and from state dicts by serializing the underlying hash map.

The `SparseDenseGrid` does a good job without an MLP in fast reconstruction tasks (e.g. RGB-D fusion, differentiable volume rendering with a decent initialization), but with an MLP, there seem no advantages in comparison to Instant-NGP as of now. Potential extensions in this line are still in progress.

### Demo: RGB-D fusion [PAMI 22]
RGB-D fusion takes in posed RGB-D images and creates colorized mesh, raw and filtered. Here, depth can either be sensor depth, or generated from a monocular depth prediction model (e.g. [omnidata](https://github.com/theNded/mini-omnidata)) with calibrated scales via [COLMAP](https://colmap.github.io/). Example datasets can be downloaded at [Google Drive](https://drive.google.com/drive/folders/12E4cTIIxmShV_ENkcvzKOQunsa0TqDVQ?usp=drive_link). Instructions for custom datasets will be available soon.

These datasets are organized by
```
- image/ # for RGB images [jpg|png]
- depth/ # for sensor depth [optional, png]
- omni_depth/ # for learned depth generated from RGB [npy]
- depth_scales.txt # calculated between learned depth and SfM
- omni_normal/ # for learned normals generated from RGB [optional, npy]
- poses.txt
- intrinsic.txt
```

To run the demo,
```sh
# Unbounded scenes, sensor depth
python demo/rgbd_fusion.py --path /path/to/dataset/samples --voxel_size 0.015 --depth_type sensor

# Bounded scenes, learned depth
python demo/rgbd_fusion.py --path /path/to/dataset/samples --resolution 512 --depth_type learned
```

### Demo: surface refinement [CVPR 23]
With learned depth, the fusion result is usually noisy. We can apply volume rendering to further optimize the shape:
```
python demo/train_scene_recon.py --path /path/to/dataset/samples --voxel_size 0.015 --depth_type learned
```
We start with a local 7x7x7 Gaussian filter to smooth the initialization.
<table>
  <tr>
  <td><img src="https://github.com/theNded/theNded.github.io/blob/master/assets/images/ash-nofilter.png" width="480"/></td>
  <td><img src="https://github.com/theNded/theNded.github.io/blob/master/assets/images/ash-filtered.png" width="480"/></td>
  </tr>
</table>

Volume rendering follows the initialization. The results will be written in `logs/datetime`. At every 500 iterations, mesh will be extracted and stored. The optimization will start with ripples on the surfaces, but finally converge to smooth reconstructions as shown above.


## API Usage
Here is a brief summary of basic usage, doc will be online soon.
### Allocation
We first initialize a 3D sparse-dense grid with 10000 sparse grid blocks. Each sparse grid contains a dense 8^3=512 array of cells, whose size is 0.01m.
```python
grid = UboundedSparseDenseGrid(in_dim=3,
                               num_embeddings=10000,
                               grid_dim=16,
                               embedding_dims=8,
                               cell_size=0.01)
```

### Initialization
We then spatially initialize the grid at input points (e.g. obtained point cloud, RGB-D scans). This results in coordinates and indices that support index-based access.
```python
with torch.no_grad():
    grid_coords, cell_coords, grid_indices, cell_indices = grid.spatial_init_(points)

    # [Optional] direct assignment
    grid.embeddings[grid_indices, cell_indices] = attributes
```

### Optimization
As a PyTorch extension, first and second-order autodiff are enabled by a differentiable query.
```python
optim = torch.optim.SGD(grid.parameters(), lr=1e-3)
for x, gt in batch:
    optim.zero_grad()
    x.requires_grad_(True)
    embedding, mask = grid(x, interpolation="linear")

    output = forward_fn(embedding, mask)

    doutput_dx = torch.autograd.grad(
        outputs=output,
        inputs=x,
        grad_outputs=torch.ones_like(output, requires_grad=False),
        create_graph=True,
        retain_graph=True)[0]

    (loss_fn(output) + grad_loss_fn(doutput_dx)).backward()
    optim.step()
```

## Milestones
- [x] Initial release
- [x] Demo: RGB-(pseudo)D SDF fusion
- [x] Demo: SDF refinement from volume rendering
- [ ] Better instructions and documentation
- [ ] Demo: LiDAR SDF fusion
- [ ] Demo: MLP integration
- [ ] CPU counterpart
