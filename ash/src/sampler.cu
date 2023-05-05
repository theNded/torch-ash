#include <c10/cuda/CUDAException.h>

#include "hashmap_gpu.cuh"
#include "minivec.h"
#include "sampler.h"

__global__ void ray_sample_kernel(stdgpu::unordered_map<MiniVec<int, 3>,
                                                        int,
                                                        MiniVecHash<int, 3>,
                                                        MiniVecEq<int, 3>> map,
                                  const MiniVec<float, 3>* ray_origins,
                                  const MiniVec<float, 3>* ray_directions,
                                  const MiniVec<float, 3>* bbox_min,
                                  const MiniVec<float, 3>* bbox_max,
                                  int* sample_count,
                                  int64_t* ray_indices,
                                  float* t_nears,
                                  float* t_fars,
                                  int64_t* prefix_sum_ray_indices,
                                  const float t_min,
                                  const float t_max,
                                  const float t_step,
                                  const float empty_space_step_multiplier) {}

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

    int len = ray_origins.size(0);
    int block_size = 1024;
    int grid_size = (len + block_size - 1) / block_size;

    auto int32_options = at::TensorOptions()
                                 .dtype(torch::kInt32)
                                 .device(ray_origins.device());
    auto int64_options = at::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(ray_origins.device());

    at::Tensor sample_counter = at::zeros({}, int32_options);
    at::Tensor prefix_sum_ray_indices = at::zeros({len}, int64_options);
    ray_sample_kernel<<<block_size, grid_size>>>(
            map, static_cast<MiniVec<float, 3>*>(ray_origins.data_ptr()),
            static_cast<MiniVec<float, 3>*>(ray_directions.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_min.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_max.data_ptr()),
            sample_counter.data_ptr<int>(),
            nullptr,  // ray_indices
            nullptr,  // t_nears
            nullptr,  // t_fars
            prefix_sum_ray_indices.data_ptr<int64_t>(), t_min, t_max, t_step,
            empty_space_step_multiplier);

    int sample_count = sample_counter.item<int>();
    auto float_options = at::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(ray_origins.device());
    at::Tensor ray_indices = at::zeros({sample_count}, int64_options);
    at::Tensor t_nears = at::zeros({sample_count}, float_options);
    at::Tensor t_fars = at::zeros({sample_count}, float_options);

    sample_counter.zero_();

    ray_sample_kernel<<<block_size, grid_size>>>(
            map, static_cast<MiniVec<float, 3>*>(ray_origins.data_ptr()),
            static_cast<MiniVec<float, 3>*>(ray_directions.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_min.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_max.data_ptr()),
            sample_counter.data_ptr<int>(),
            ray_indices.data_ptr<int64_t>(),  // ray_indices
            t_nears.data_ptr<float>(),        // t_nears
            t_fars.data_ptr<float>(),         // t_fars
            prefix_sum_ray_indices.data_ptr<int64_t>(), t_min, t_max, t_step,
            empty_space_step_multiplier);

    return std::make_tuple(ray_indices, t_nears, t_fars,
                           prefix_sum_ray_indices);
}
