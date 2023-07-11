#include <c10/cuda/CUDAException.h>

#include "hashmap_gpu.cuh"
#include "minivec.h"
#include "sampler.h"

__global__ void ray_find_near_far_kernel(
        stdgpu::unordered_map<MiniVec<int, 3>,
                              int,
                              MiniVecHash<int, 3>,
                              MiniVecEq<int, 3>> map,
        const MiniVec<float, 3>* ray_origins,
        const MiniVec<float, 3>* ray_directions,
        const MiniVec<float, 3>* bbox_min,
        const MiniVec<float, 3>* bbox_max,
        float* ray_nears,
        float* ray_fars,
        const float t_min,
        const float t_max,
        const float t_step,
        const float grid2cell_multiplier,
        const int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    auto xyz_min = bbox_min[0];
    auto xyz_max = bbox_max[0];

    auto ray_o = ray_origins[idx];
    auto ray_d = ray_directions[idx];

    const float cell2grid_multiplier = 1.0 / grid2cell_multiplier;

    const float t_cell_step = t_step;
    const float t_grid_step = t_step * grid2cell_multiplier * 0.5;

    float near = -1;
    float far = t_max;
    float t = t_min;

    // Coarse estimate
    while (t < t_max) {
        auto xyz_cell = ray_o + t * ray_d;

        // Bound check
        bool in_bound = true;
        for (int d = 0; d < 3; ++d) {
            in_bound = in_bound &&
                       (xyz_cell[d] >= xyz_min[d] && xyz_cell[d] <= xyz_max[d]);
        }
        if (!in_bound) break;

        // Empty check
        auto xyz_grid = floor((xyz_cell * cell2grid_multiplier)).cast<int>();
        auto it = map.find(xyz_grid);
        if (it == map.end()) {
            t += t_grid_step;
            continue;
        }

        // Valid sample exists -- assign near
        // If this line is never reached, mark near as -1, indicating no
        // valid sample exists along this ray.
        near = (near < 0) ? t : max(near - t_grid_step, t_min);
        far = min(t + t_grid_step, t_max);

        t += t_grid_step;
    }
    ray_nears[idx] = near;
    ray_fars[idx] = far;
}

std::tuple<at::Tensor, at::Tensor> ray_find_near_far(
        const HashMap& hashmap,
        const at::Tensor& ray_origins,
        const at::Tensor& ray_directions,
        const at::Tensor& bbox_min,
        const at::Tensor& bbox_max,
        const float t_min,
        const float t_max,
        const float t_step,
        const float grid2cell_multiplier) {
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

    int len = ray_origins.size(0);
    int block_size = 1024;
    int grid_size = (len + block_size - 1) / block_size;

    auto float_options = at::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(ray_origins.device());

    at::Tensor ray_nears = at::zeros({len}, float_options);
    at::Tensor ray_fars = at::zeros({len}, float_options);
    ray_find_near_far_kernel<<<block_size, grid_size>>>(
            map, static_cast<MiniVec<float, 3>*>(ray_origins.data_ptr()),
            static_cast<MiniVec<float, 3>*>(ray_directions.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_min.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_max.data_ptr()),
            ray_nears.data_ptr<float>(), ray_fars.data_ptr<float>(), t_min,
            t_max, t_step, grid2cell_multiplier, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_tuple(ray_nears, ray_fars);
}

__global__ void ray_sample_count_kernel(
        stdgpu::unordered_map<MiniVec<int, 3>,
                              int,
                              MiniVecHash<int, 3>,
                              MiniVecEq<int, 3>> map,
        const MiniVec<float, 3>* ray_origins,
        const MiniVec<float, 3>* ray_directions,
        const MiniVec<float, 3>* bbox_min,
        const MiniVec<float, 3>* bbox_max,
        int64_t* ray_sample_counts,  // temporary
        int64_t* ray_indices,
        float* t_nears,
        float* t_fars,
        const int64_t* prefix_sum_ray_sample_counts,
        const float t_min,
        const float t_max,
        const float t_step,
        const float grid2cell_multiplier,
        const int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }

    auto xyz_min = bbox_min[0];
    auto xyz_max = bbox_max[0];

    auto ray_o = ray_origins[idx];
    auto ray_d = ray_directions[idx];

    const float cell2grid_multiplier = 1.0 / grid2cell_multiplier;

    const float t_cell_step = t_step;
    const float t_grid_step = t_step * grid2cell_multiplier;

    float t = t_min;
    float t_near = t_min;
    int local_count = 0;

    bool allocation_pass = (ray_sample_counts != nullptr);
    bool assignment_pass = (ray_indices != nullptr) && (t_nears != nullptr) &&
                           (t_fars != nullptr) &&
                           (prefix_sum_ray_sample_counts != nullptr);
    if (!allocation_pass && !assignment_pass) return;
    if (allocation_pass && assignment_pass) {
        printf("Cannot run allocation and assignment in the same pass, "
               "abort!\n");
        return;
    }

    int start = -1, end = -1;
    if (assignment_pass) {
        start = (idx == 0) ? 0 : prefix_sum_ray_sample_counts[idx - 1];
        end = prefix_sum_ray_sample_counts[idx];
    }

    while (t < t_max) {
        auto xyz_cell = ray_o + t * ray_d;

        // Bound check
        bool in_bound = true;
        for (int d = 0; d < 3; ++d) {
            in_bound = in_bound &&
                       (xyz_cell[d] >= xyz_min[d] && xyz_cell[d] <= xyz_max[d]);
        }
        if (!in_bound) break;

        // Empty check
        auto xyz_grid = floor((xyz_cell * cell2grid_multiplier)).cast<int>();
        auto it = map.find(xyz_grid);
        if (it == map.end()) {
            t += t_grid_step;
        } else {
            t += t_cell_step;
            if (assignment_pass) {
                ray_indices[start + local_count] = idx;
                t_nears[start + local_count] = t_near;
                t_fars[start + local_count] = t;
            }
            t_near = t;

            local_count++;
        }
    }
    if (allocation_pass) {
        ray_sample_counts[idx] = local_count;
    }

    if (assignment_pass) {
        if (end != start + local_count) {
            printf("[thread %d]Inconsistent sample counts between allocation "
                   "and assignment. Should never reach here!\n",
                   idx);
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ray_sample(
        const HashMap& hashmap,
        const at::Tensor& ray_origins,
        const at::Tensor& ray_directions,
        const at::Tensor& bbox_min,
        const at::Tensor& bbox_max,
        const float t_min,
        const float t_max,
        const float t_step,
        const float grid2cell_multiplier) {
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

    int len = ray_origins.size(0);
    int block_size = 1024;
    int grid_size = (len + block_size - 1) / block_size;

    auto int32_options = at::TensorOptions()
                                 .dtype(torch::kInt32)
                                 .device(ray_origins.device());
    auto int64_options = at::TensorOptions()
                                 .dtype(torch::kInt64)
                                 .device(ray_origins.device());
    auto float_options = at::TensorOptions()
                                 .dtype(torch::kFloat32)
                                 .device(ray_origins.device());

    at::Tensor ray_sample_counts = at::zeros({len}, int64_options);
    ray_sample_count_kernel<<<block_size, grid_size>>>(
            map, static_cast<MiniVec<float, 3>*>(ray_origins.data_ptr()),
            static_cast<MiniVec<float, 3>*>(ray_directions.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_min.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_max.data_ptr()),
            ray_sample_counts.data_ptr<int64_t>(), nullptr, nullptr, nullptr,
            nullptr, t_min, t_max, t_step, grid2cell_multiplier, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    // std::cout << ray_sample_counts << "\n";
    at::Tensor prefix_sum_ray_sample_counts = at::cumsum(ray_sample_counts, 0);
    // std::cout << prefix_sum_ray_sample_counts << "\n";

    int64_t total_count =
            prefix_sum_ray_sample_counts.index({-1}).item<int64_t>();
    // std::cout << "total samples: " << total_count;

    at::Tensor ray_indices = at::zeros({total_count}, int64_options);
    at::Tensor t_nears = at::zeros({total_count}, float_options);
    at::Tensor t_fars = at::zeros({total_count}, float_options);

    ray_sample_count_kernel<<<block_size, grid_size>>>(
            map, static_cast<MiniVec<float, 3>*>(ray_origins.data_ptr()),
            static_cast<MiniVec<float, 3>*>(ray_directions.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_min.data_ptr()),
            static_cast<MiniVec<float, 3>*>(bbox_max.data_ptr()), nullptr,
            ray_indices.data_ptr<int64_t>(), t_nears.data_ptr<float>(),
            t_fars.data_ptr<float>(),
            prefix_sum_ray_sample_counts.data_ptr<int64_t>(), t_min, t_max,
            t_step, grid2cell_multiplier, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_tuple(ray_indices, t_nears, t_fars,
                           prefix_sum_ray_sample_counts);
}
