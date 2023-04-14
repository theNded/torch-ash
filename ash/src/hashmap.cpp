#include "hashmap.h"

#include <iostream>
#include <stdexcept>

#include "hashmap_impl.h"

HashMap::HashMap(int64_t key_dim,
                 int64_t capacity,
                 at::Tensor& heap,
                 c10::Device& device) {
    if (device == at::kCPU) {
        impl_ = create_hashmap_cpu_impl(key_dim, capacity, heap, device);
    } else {
        impl_ = create_hashmap_gpu_impl(key_dim, capacity, heap, device);
    }
}

std::pair<at::Tensor, at::Tensor> HashMap::find(const at::Tensor& keys) {
    return impl_->find(keys);
}

void HashMap::insert_keys(const at::Tensor& keys) {
    return impl_->insert_keys(keys);
}

void HashMap::insert(
        const at::Tensor& keys,
        const std::unordered_map<std::string, at::Tensor>& values,
        const std::unordered_map<std::string, at::Tensor>& external_values) {
    return impl_->insert(keys, values, external_values);
}

void HashMap::erase(const at::Tensor& keys) { return impl_->erase(keys); }

void HashMap::clear() { return impl_->clear(); }

void HashMap::load_states(
        const std::unordered_map<std::string, at::Tensor>& states) {
    impl_->load_states(states);
}

std::pair<at::Tensor, at::Tensor> HashMap::items() { return impl_->items(); }

int64_t HashMap::size() { return impl_->size(); }

at::Device HashMap::device() { return impl_->device(); }
