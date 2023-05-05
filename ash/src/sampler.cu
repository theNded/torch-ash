#include <c10/cuda/CUDAException.h>

#include "hashmap_gpu.cuh"
#include "minivec.h"
#include "sampler.h"

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ray_sample(
        const HashMap& hashmap,
        const at::Tensor& ray_origins,
        const at::Tensor& ray_directions,
        const at::Tensor& bbox_min,
        const at::Tensor& bbox_max,
        const float t_min,
        const float t_max,
        const float t_step,
        const float empty_space_step_multiplier) {
    // Ray sample is for 3D hash map only
    using key_t = MiniVec<int, 3>;
    using eq_t = MiniVecEq<int, 3>;
    using hash_t = MiniVecHash<int, 3>;

    auto hashmap_impl =
            std::dynamic_pointer_cast<HashMapGPUImpl<key_t, hash_t, eq_t>>(
                    hashmap.impl_);
    if (hashmap_impl == nullptr) {
        AT_ERROR(
                "hashmap is not a GPU hashmap of type <int, 3>, ray sample not "
                "supported");
    }

    stdgpu::unordered_map<key_t, int, hash_t, eq_t> map =
            hashmap_impl->backend_;
    std::cout << "stdgpu map size: " << map.size() << std::endl;

    return std::make_tuple(at::zeros({0}, ray_origins.options()),
                           at::zeros({0}, ray_origins.options()),
                           at::zeros({0}, ray_origins.options()),
                           at::zeros({0}, ray_origins.options()));
}
