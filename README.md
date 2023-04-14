# Neural Indoor Scene Recon

## Install in an external project
```
pip install -U git+ssh://git@github.com/theNded/torch-ash
```

## Config
Configuration for init_recon:
```
conda create -n ash python=3.8 pytorch torchvision -c pytorch
pip install -r requirements_init.txt
pip install --verbose .
```

## Preprocessing

```
git submodule update --init --recursive
pip install -r requirements_pre.txt
```

## Backend
`torch-ash`, consisting of spatially hashed voxel blocks. 

Dependency: 
- `stdgpu` included in the `ext/` folder. Modified to allow a longer chain to avoid collisions.

## Preprocess
Use `./scripts/preprocess.sh path/to/scenes` to preprocess a scene.

The dataset has been preprocessed: https://drive.google.com/file/d/1Ybb7JagMj-Yhq6Zv2UuH-gmnzsyF8Zct/view?usp=sharing.
There is no need to run the preprocess script.

For each scene, the directory is
```
scene_name
|__ samples
    |__ depth  <- GT high res depth, for reference only
    |__ image  <- GT high res image, for reference only
    |__ intrinsic_depth.txt <- GT highres depth intrinsics
    |__ intrinsic_color.txt <- GT highres color intrinsics
    |__ poses.txt <- GT poses
    |
    |__ omni_image  <- center cropped gt images. Loaded in the dataloader
    |__ omni_depth  <- depth predicted by omnidata. Loaded in the dataloader
    |__ omni_normal <- normal predicted by omnidata. Loaded in the dataloader
    |__ intrinsic_omni.txt <- center cropped image intrinsic
    |
    |__ colmap <- for sparse recon, already done in preprocessing
    |__ omni2gt.npz <- scale factors used to align gt (metric), colmap, and omnidata scale
    |
    |__ custom_labels.txt <- labels manually generated to fed into lseg
    |__ label             <- labels predicted by lseg. (optionally) loaded in the dataloader

```

## Initialization
Use `./scripts/init_recon.sh /path/to/scenes` to generate initial grids via TSDF integration, and extract point clouds. 
They will be written to `runs/$scene_name`.

## Refinement
Use `./scripts/train.sh /path/to/scenes` to train.

## Evaluation
Use `./scripts/eval.sh /path/to/scenes` to get MarchingCubes, TSDF-refusion, and rendering.
