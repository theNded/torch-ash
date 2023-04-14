#include "hashmap_cpu.hpp"

std::shared_ptr<HashMapImpl> create_hashmap_cpu_impl(int64_t key_dim,
                                                     int64_t capacity,
                                                     at::Tensor& heap,
                                                     c10::Device& device) {
    std::shared_ptr<HashMapImpl> impl_ptr = nullptr;
    throw std::runtime_error("Not implemented");
    return impl_ptr;
}
