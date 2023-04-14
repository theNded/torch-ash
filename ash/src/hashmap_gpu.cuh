#include <c10/cuda/CUDAException.h>
#include <thrust/device_vector.h>

#include <stdgpu/unordered_map.cuh>

#include "dispatch.h"
#include "hashmap_impl.h"

struct HeapContext {
    int* heap_counter;
    int* heap;
};

// External to avoid constructor on kernel
HeapContext construct_heap_ctx(at::Tensor& heap_counter, at::Tensor& heap) {
    HeapContext ctx;
    ctx.heap_counter = heap_counter.data_ptr<int>();
    ctx.heap = heap.data_ptr<int>();
    return ctx;
}

template <typename Blob>
struct ValueContext {
    /// Reinterpret values as blocks for faster copying

    /// value_ptrs[i]: Blob[M][value_blob_sizes[i]]
    /// external_value_ptrs[i]: Blob[M][value_blob_sizes[i]]
    int64_t n_values;            // N
    int64_t* value_blob_sizes;   // [N]
    Blob** value_ptrs;           // [N]
    Blob** external_value_ptrs;  // [N]
};

template <typename Blob>
ValueContext<Blob> construct_value_ctx(
        const std::unordered_map<std::string, at::Tensor>& values,
        const std::unordered_map<std::string, at::Tensor>& external_values,
        const std::vector<int64_t>& value_blob_sizes) {
    // Use DISPATCH to decide the Blob template before calling this function
    ValueContext<Blob> ctx;

    ctx.n_values = value_blob_sizes.size();

    C10_CUDA_CHECK(cudaMalloc(&ctx.value_ptrs, ctx.n_values * sizeof(Blob*)));
    C10_CUDA_CHECK(
            cudaMalloc(&ctx.external_value_ptrs, ctx.n_values * sizeof(Blob*)));

    thrust::device_vector<int64_t> blob_size_vec = value_blob_sizes;
    ctx.value_blob_sizes = thrust::raw_pointer_cast(blob_size_vec.data());

    std::vector<Blob*> value_ptrs, external_value_ptrs;
    for (auto kv : values) {
        auto& name = kv.first;
        auto& tensor = kv.second;
        auto& external_tensor = external_values.at(name);

        value_ptrs.push_back((Blob*)tensor.data_ptr());
        external_value_ptrs.push_back((Blob*)external_tensor.data_ptr());
    }

    thrust::device_ptr<Blob*> ctx_value_ptr(ctx.value_ptrs);
    thrust::copy(value_ptrs.begin(), value_ptrs.end(), ctx_value_ptr);

    thrust::device_ptr<Blob*> ctx_external_value_ptr(ctx.external_value_ptrs);
    thrust::copy(external_value_ptrs.begin(), external_value_ptrs.end(),
                 ctx_external_value_ptr);

    return ctx;
}

template <typename Blob>
void destruct_value_ctx(ValueContext<Blob> ctx) {
    C10_CUDA_CHECK(cudaFree(ctx.value_ptrs));
    C10_CUDA_CHECK(cudaFree(ctx.external_value_ptrs));
}

// Compute the max data blob size for fast copying/dispatching.
std::pair<int64_t, std::vector<int64_t>> get_max_blob_size(
        const std::unordered_map<std::string, at::Tensor>& values,
        const std::unordered_map<std::string, at::Tensor>& external_values) {
    std::vector<int64_t> value_item_sizes;
    for (auto& kv : values) {
        auto& name = kv.first;
        auto& tensor = kv.second;
        auto& external_tensor = external_values.at(name);

        int64_t item_size =
                (tensor.numel() / tensor.size(0)) * tensor.element_size();
        value_item_sizes.push_back(item_size);
    }

    int64_t block_size = 1u;

    int64_t n = value_item_sizes.size();
    const std::vector<int64_t> kDivisors = {16, 12, 8, 4, 2, 1};
    for (const auto& divisor : kDivisors) {
        bool valid = true;

        for (int64_t i = 0; i < n; ++i) {
            int64_t value_item_size = value_item_sizes[i];
            valid = valid && (value_item_size % divisor == 0);
        }
        if (valid) {
            block_size = divisor;
            break;
        }
    }

    for (auto& s : value_item_sizes) {
        s /= block_size;
    }
    return std::make_pair(block_size, value_item_sizes /* in blobs */);
}

template <typename Key, typename Hash, typename Eq>
class HashMapGPUImpl : public HashMapImpl {
public:
    HashMapGPUImpl(int64_t key_dim,
                   int64_t capacity,
                   at::Tensor& heap,
                   c10::Device device)
        : HashMapImpl(key_dim, capacity, heap, device) {
        allocate(capacity_);
    }

    ~HashMapGPUImpl() { free(); }

    void allocate(int64_t capacity) {
        backend_ =
                stdgpu::unordered_map<Key, int, Hash, Eq>::createDeviceObject(
                        capacity);
    }
    void free() {
        stdgpu::unordered_map<Key, int, Hash, Eq>::destroyDeviceObject(
                backend_);
    }

    std::pair<at::Tensor, at::Tensor> find(const at::Tensor& keys) override;

    void insert_keys(const at::Tensor& keys) override;

    void insert(const at::Tensor& keys,
                const std::unordered_map<std::string, at::Tensor>& values,
                const std::unordered_map<std::string, at::Tensor>&
                        external_values) override;

    void erase(const at::Tensor& keys) override;

    void clear() override;

    void load_states(
            const std::unordered_map<std::string, at::Tensor>& states) override;

    std::pair<at::Tensor, at::Tensor> items() override;

    int64_t size() override { return backend_.size(); }

    stdgpu::unordered_map<Key, int, Hash, Eq> backend_;
};

// Find
template <typename Key, typename Hash, typename Eq>
__global__ void find_kernel(stdgpu::unordered_map<Key, int, Hash, Eq> map,
                            const Key* keys,
                            int64_t* indices,
                            bool* masks,
                            int count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = keys[tid];
    auto iter = map.find(key);
    bool flag = (iter != map.end());

    masks[tid] = flag;
    indices[tid] = flag ? iter->second : 0;
}

template <typename Key, typename Hash, typename Eq>
std::pair<at::Tensor, at::Tensor> HashMapGPUImpl<Key, Hash, Eq>::find(
        const at::Tensor& keys) {
    int count = keys.size(0);

    // Return
    auto index_options =
            at::TensorOptions().dtype(torch::kInt64).device(keys.device());
    auto mask_options =
            at::TensorOptions().dtype(torch::kBool).device(keys.device());

    at::Tensor indices = at::zeros({count}, index_options);
    at::Tensor masks = at::zeros({count}, mask_options);

    int threads = 128;
    int blocks = (count + threads - 1) / threads;

    auto keys_ptr = static_cast<const Key*>(keys.data_ptr());
    auto indices_ptr = indices.data_ptr<int64_t>();
    auto masks_ptr = masks.data_ptr<bool>();

    find_kernel<Key, Hash, Eq>
            <<<blocks, threads>>>(backend_, keys_ptr,      // input
                                  indices_ptr, masks_ptr,  // output
                                  count);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_pair(indices, masks);
}

// insert -- no keys
template <typename Key, typename Hash, typename Eq>
__global__ void insert_kernel(stdgpu::unordered_map<Key, int, Hash, Eq> map,
                              const Key* keys,
                              HeapContext ctx,
                              int count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = keys[tid];

    // First apply 'try insert' with a dummy index
    auto res = map.emplace(key, 0);

    // If success, change the iterator and provide the actual index
    if (res.second) {
        int top = atomicAdd(ctx.heap_counter, 1);
        int idx = ctx.heap[top];

        // Update from the dummy index
        res.first->second = idx;
    }
}

template <typename Key, typename Hash, typename Eq>
void HashMapGPUImpl<Key, Hash, Eq>::insert_keys(const at::Tensor& keys) {
    int count = keys.size(0);

    int threads = 128;
    int blocks = (count + threads - 1) / threads;

    auto heap_ctx = construct_heap_ctx(heap_counter_, heap_);
    auto keys_ptr = static_cast<const Key*>(keys.data_ptr());

    insert_kernel<Key, Hash, Eq>
            <<<blocks, threads>>>(backend_, keys_ptr, heap_ctx, count);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename Key, typename Hash, typename Eq, typename blob_t>
__global__ void insert_kernel(stdgpu::unordered_map<Key, int, Hash, Eq> map,
                              const Key* keys,
                              HeapContext heap_ctx,
                              ValueContext<blob_t> value_ctx,
                              int count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = keys[tid];

    // First apply 'try insert' with a dummy index
    auto res = map.emplace(key, 0);

    // If success, change the iterator and provide the actual index
    if (res.second) {
        int top = atomicAdd(heap_ctx.heap_counter, 1);
        int idx = heap_ctx.heap[top];

        // Copy/reset non-templated value in buffer
        for (int j = 0; j < value_ctx.n_values; ++j) {
            const int64_t blob_size = value_ctx.value_blob_sizes[j];

            blob_t* dst_ptr =
                    value_ctx.external_value_ptrs[j] + blob_size * idx;
            blob_t* src_ptr = value_ctx.value_ptrs[j] + blob_size * tid;
            for (int b = 0; b < blob_size; ++b) {
                dst_ptr[b] = src_ptr[b];
            }
        }

        // Update from the dummy index
        res.first->second = idx;
    }
}

template <typename Key, typename Hash, typename Eq>
void HashMapGPUImpl<Key, Hash, Eq>::insert(
        const at::Tensor& keys,
        const std::unordered_map<std::string, at::Tensor>& values,
        const std::unordered_map<std::string, at::Tensor>& external_values) {
    int count = keys.size(0);

    int threads = 128;
    int blocks = (count + threads - 1) / threads;

    auto heap_ctx = construct_heap_ctx(heap_counter_, heap_);

    auto result = get_max_blob_size(values, external_values);
    int64_t shared_blob_size = result.first;
    std::vector<int64_t> value_blob_sizes = result.second;

    auto keys_ptr = static_cast<const Key*>(keys.data_ptr());
    DISPATCH_DIVISOR_SIZE_TO_BLOB_T(shared_blob_size, [&]() {
        auto value_ctx = construct_value_ctx<blob_t>(values, external_values,
                                                     value_blob_sizes);
        insert_kernel<Key, Hash, Eq, blob_t>
                <<<blocks, threads>>>(backend_, keys_ptr,   // input
                                      heap_ctx, value_ctx,  // output
                                      count);
        C10_CUDA_CHECK(cudaDeviceSynchronize());

        destruct_value_ctx<blob_t>(value_ctx);
    });
}

// Erase
template <typename Key, typename Hash, typename Eq>
__global__ void erase_kernel(stdgpu::unordered_map<Key, int, Hash, Eq> map,
                             const Key* keys,
                             HeapContext ctx,
                             int count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = keys[tid];
    auto iter = map.find(key);
    bool flag = (iter != map.end());

    if (flag) {  // found
        flag = map.erase(key);
        if (flag) {  // success (could fail if duplicates are erased
                     // simultaneously)
            int top = atomicSub(ctx.heap_counter, 1);
            ctx.heap[top - 1] = iter->second;
        }
    }
}

template <typename Key, typename Hash, typename Eq>
void HashMapGPUImpl<Key, Hash, Eq>::erase(const at::Tensor& keys) {
    int count = keys.size(0);

    int threads = 128;
    int blocks = (count + threads - 1) / threads;

    auto ctx = construct_heap_ctx(heap_counter_, heap_);
    auto keys_ptr = static_cast<const Key*>(keys.data_ptr());

    erase_kernel<Key, Hash, Eq>
            <<<blocks, threads>>>(backend_, keys_ptr, ctx, count);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
}

// Clear
template <typename Key, typename Hash, typename Eq>
void HashMapGPUImpl<Key, Hash, Eq>::clear() {
    backend_.clear();
    reset_heap();
}

// Enumerate
template <typename Key>
struct IndexExtractor {
    __host__ __device__ int64_t
    operator()(const thrust::pair<Key, int>& x) const {
        return int64_t(x.second);
    }
};

template <typename Key>
struct KeyExtractor {
    __host__ __device__ Key operator()(const thrust::pair<Key, int>& x) const {
        return x.first;
    }
};

template <typename Key, typename Hash, typename Eq>
std::pair<at::Tensor, at::Tensor> HashMapGPUImpl<Key, Hash, Eq>::items() {
    int64_t n = backend_.size();
    if (n == 0) {
        std::cout << "No active entries found.\n";
        return std::make_pair(at::Tensor(), at::Tensor());
    }

    std::vector<int64_t> key_shape{n, (int64_t)key_dim_};

    auto range = backend_.device_range();
    auto key_option = at::TensorOptions()
                              .dtype(torch::kInt32)
                              .device(heap_counter_.device());
    at::Tensor active_keys = at::zeros(key_shape, key_option);

    auto index_option = at::TensorOptions()
                                .dtype(torch::kInt64)
                                .device(heap_counter_.device());
    at::Tensor active_indices = at::zeros({n}, index_option);

    thrust::transform(range.begin(), range.end(),
                      static_cast<Key*>(active_keys.data_ptr()),
                      KeyExtractor<Key>());
    thrust::transform(range.begin(), range.end(),
                      static_cast<int64_t*>(active_indices.data_ptr()),
                      IndexExtractor<Key>());

    return std::make_pair(active_keys, active_indices);
}

template <typename Key, typename Hash, typename Eq>
__global__ void load_states_kernel(
        stdgpu::unordered_map<Key, int, Hash, Eq> map,
        const Key* keys,
        int64_t* indices,
        int count) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= count) return;

    Key key = keys[tid];
    int index = indices[tid];
    auto res = map.emplace(key, index);
    if (!res.second) {
        printf("Error loading, should never reach here!\n");
    }
}

template <typename Key, typename Hash, typename Eq>
void HashMapGPUImpl<Key, Hash, Eq>::load_states(
        const std::unordered_map<std::string, at::Tensor>& states) {
    using namespace torch::indexing;

    auto active_keys = states.at("active_keys");
    auto active_indices = states.at("active_indices");
    auto heap = states.at("heap");

    // Load heap info
    int count = active_keys.size(0);
    heap_counter_[0] = count;
    heap_.index_put_({None}, heap.to(heap_.device()));

    if (count == 0) return;

    // Load key-index map into the unordered map
    int threads = 128;
    int blocks = (count + threads - 1) / threads;

    auto keys_ptr = static_cast<const Key*>(active_keys.data_ptr());
    auto indices_ptr = active_indices.data_ptr<int64_t>();

    load_states_kernel<Key, Hash, Eq>
            <<<blocks, threads>>>(backend_, keys_ptr, indices_ptr, count);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
}
