#pragma once

#include <torch/extension.h>

#include "hashmap_impl.h"

std::shared_ptr<HashMapImpl> create_hashmap_gpu_impl(int64_t key_dim,
                                                     int64_t capacity,
                                                     at::Tensor& heap,
                                                     c10::Device& device);

std::shared_ptr<HashMapImpl> create_hashmap_cpu_impl(int64_t key_dim,
                                                     int64_t capacity,
                                                     at::Tensor& heap,
                                                     c10::Device& device);

// Expose to the interface
class HashMap {
public:
    // Construction from scratch
    // Only maintain the very core. heap, values are all maintained outside.
    HashMap(int64_t key_dim,
            int64_t capacity,
            at::Tensor& heap,
            c10::Device& device);

    // Find (indices, masks) for given entries
    // Indices: index to the associated embeddings/values
    // Masks: key found or not
    std::pair<at::Tensor, at::Tensor> find(const at::Tensor& keys);

    // Activate key entries without inserting values
    void insert_keys(const at::Tensor& keys);

    // Insert key entries with value(s)
    void insert(
            const at::Tensor& keys,
            const std::unordered_map<std::string, at::Tensor>& values,
            const std::unordered_map<std::string, at::Tensor>& external_values);

    // Erase key entries
    void erase(const at::Tensor& keys);

    // Clear the keys and reset the heap
    void clear();

    // Load the keys and heaps into the internal data structure by
    // in-place modification
    // > heap
    // > <key, index> map
    // <keys, index> map are from items()
    // Values are explicitly loaded to tensor lists
    void load_states(const std::unordered_map<std::string, at::Tensor>& states);

    // Return active_keys and active_indices
    std::pair<at::Tensor, at::Tensor> items();
    int64_t size();
    at::Device device();

    std::shared_ptr<HashMapImpl> impl_;
};
