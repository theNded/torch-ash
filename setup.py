from setuptools import setup
import subprocess
import os
import os.path as osp
import warnings

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = "0.0.1"

INSTALL_REQUIREMENTS = ["wandb", "configargparse"]
include_dirs = [
    osp.join(ROOT_DIR, "src"),
]

CC_FLAGS = ["-std=c++17", "-Wunused-local-typedefs"]
NVCC_FLAGS = ["-std=c++17"]

CC_FLAGS += ["-O3", "-fPIC"]
NVCC_FLAGS += ["-O3", "-Xcompiler=-fno-gnu-unique,-fPIC"]

# Build dependency with Cmake
stdgpu_dir = osp.join(ROOT_DIR, "ext", "stdgpu")
stdgpu_build_dir = osp.join(stdgpu_dir, "build")
stdgpu_install_dir = osp.join(stdgpu_dir, "install")

cmake_flags = [
    f"-DCMAKE_INSTALL_PREFIX={stdgpu_install_dir}",
    "-DSTDGPU_BUILD_SHARED_LIBS=OFF",
    "-DSTDGPU_BUILD_EXAMPLES=OFF",
    "-DSTDGPU_BUILD_TESTS=OFF",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
]

print("Building external project ...")
subprocess.run(["cmake", "-S", stdgpu_dir, "-B", stdgpu_build_dir] + cmake_flags)
subprocess.run(["cmake", "--build", stdgpu_build_dir, "-t", "install"])

include_dirs.append(osp.join(stdgpu_install_dir, "include/"))
library_dirs = [osp.join(stdgpu_install_dir, "lib")]
libraries = ["stdgpu"]

# From PyTorch3D
cub_home = os.environ.get("CUB_HOME", None)
if cub_home is None:
    prefix = os.environ.get("CONDA_PREFIX", None)
    if prefix is not None and os.path.isdir(prefix + "/include/cub"):
        cub_home = prefix + "/include"

if cub_home is None:
    warnings.warn(
        "The environment variable `CUB_HOME` was not found."
        "Installation will fail if your system CUDA toolkit version is less than 11."
        "NVIDIA CUB can be downloaded "
        "from `https://github.com/NVIDIA/cub/releases`. You can unpack "
        "it to a location of your choice and set the environment variable "
        "`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
    )
else:
    include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))

try:
    ext_modules = [
        CUDAExtension(
            "__ash",
            [
                "ash/src/pybind.cpp",
                "ash/src/hashmap.cpp",
                "ash/src/hashmap_gpu.cu",
                "ash/src/hashmap_cpu.cpp",
                "ash/src/sampler.cu",
                "ash/src/sparsedense_grid.cu"
            ],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args={
                "cxx": CC_FLAGS,
                "nvcc": NVCC_FLAGS,
            },
            optional=False,
        ),
    ]
except:
    import warnings

    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

# shutil.copy(osp.join(library_dirs[0], 'libstdgpu.so'),
#             osp.join('/home', 'wei', 'libstdgpu.so'))

setup(
    name="ash",
    version=__version__,
    author="Wei Dong",
    author_email="weidong@andrew.cmu.edu",
    description="",
    long_description="",
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.5.0"],
    packages=["ash"],  # Directory name
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
