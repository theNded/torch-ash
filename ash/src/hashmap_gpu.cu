#include <stdgpu/unordered_map.cuh>

#include "hashmap_gpu.cuh"

std::shared_ptr<HashMapImpl> create_hashmap_gpu_impl(int64_t key_dim,
                                                     int64_t capacity,
                                                     at::Tensor& heap,
                                                     c10::Device& device) {
    std::shared_ptr<HashMapImpl> impl_ptr = nullptr;

    DISPATCH_DTYPE_AND_DIM_TO_TEMPLATE(torch::kInt32, key_dim, [&] {
        // make the compiler happy
        impl_ptr = std::make_shared<HashMapGPUImpl<key_t, hash_t, eq_t>>(
                key_dim, capacity, heap, device);
    });
    return impl_ptr;
}
