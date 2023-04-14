#pragma once

#include <torch/extension.h>

#include <cstdio>
#include <memory>

class HashMapImpl {
public:
    // From construction
    HashMapImpl(int64_t key_dim,
                int64_t capacity,
                at::Tensor& heap,
                c10::Device& device) {
        key_dim_ = key_dim;
        capacity_ = capacity;

        // Heap counter
        auto heap_option =
                at::TensorOptions().dtype(torch::kInt32).device(device);
        heap_counter_ = at::zeros({1}, heap_option);

        heap_ = heap;
    }

    void reset_heap() {
        using namespace torch::indexing;
        heap_counter_.index_put_({0}, 0);
        heap_.index_put_({None}, at::arange(0, capacity_,
                                            at::TensorOptions()
                                                    .dtype(torch::kInt32)
                                                    .device(heap_.device())));
    }

    virtual std::pair<at::Tensor, at::Tensor> find(const at::Tensor& keys) = 0;
    virtual void insert_keys(const at::Tensor&) = 0;

    virtual void insert(
            const at::Tensor& keys,
            const std::unordered_map<std::string, at::Tensor>& values,
            const std::unordered_map<std::string, at::Tensor>&
                    external_values) = 0;

    virtual void erase(const at::Tensor& keys) = 0;

    virtual void clear() = 0;

    virtual void load_states(
            const std::unordered_map<std::string, at::Tensor>& states) = 0;

    virtual std::pair<at::Tensor, at::Tensor> items() = 0;

    virtual int64_t size() = 0;

    at::Device device() { return heap_counter_.device(); }

    int64_t key_dim_;
    int64_t capacity_;

    at::Tensor heap_counter_;
    at::Tensor heap_;
};
