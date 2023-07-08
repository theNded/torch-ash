#include <c10/cuda/CUDAException.h>

#include "grid.h"
#include "mc_macros.h"
#include "minivec.h"

template <typename scalar_t>
class SmoothStepFunctor {
public:
    using scalar_type = scalar_t;
    __device__ __inline__ scalar_t operator()(const scalar_t& x) const {
        return x * x * (3 - 2 * x);
    }
};

template <typename scalar_t>
class DiffSmoothStepFunctor {
public:
    using scalar_type = scalar_t;
    __device__ __inline__ scalar_t operator()(const scalar_t& x) const {
        return 6 * x * (1 - x);
    }
};

template <typename scalar_t>
class LinearFunctor {
public:
    using scalar_type = scalar_t;
    __device__ __inline__ scalar_t operator()(const scalar_t& x) const {
        return x;
    }
};

template <typename scalar_t>
class DiffLinearFunctor {
public:
    using scalar_type = scalar_t;
    __device__ __inline__ scalar_t operator()(const scalar_t& x) const {
        return 1;
    }
};

template <typename Functor>
struct functor_t {
    __host__ __device__ typename Functor::scalar_type operator()(
            const typename Functor::scalar_type& x) const {
        return Functor()(x);
    }
};

#define DISPATCH_INTERP_FUNCTOR(name, ...)                         \
    if (name == "linear") {                                        \
        using InterpFunctor = LinearFunctor<scalar_t>;             \
        using DiffInterpFunctor = DiffLinearFunctor<scalar_t>;     \
        return __VA_ARGS__();                                      \
    } else if (name == "smooth_step") {                            \
        using InterpFunctor = SmoothStepFunctor<scalar_t>;         \
        using DiffInterpFunctor = DiffSmoothStepFunctor<scalar_t>; \
        return __VA_ARGS__();                                      \
    } else {                                                       \
        AT_ERROR("Unknown interpolation functor: ", name);         \
    }

const float kInterpSumWeightThreshold = 0.99999;

// Now only dispatch dtypes, all the queries are for 3D
// TODO: dispatch for 2D/4D later
template <typename InterpFunctor, typename scalar_t>
__global__ void query_forward_kernel(
        const scalar_t* __restrict__ embeddings,
        const MiniVec<float, 3>* __restrict__ offsets,
        const int64_t* __restrict__ grid_indices,
        const int64_t* __restrict__ cell_indices,
        const bool* __restrict__ masks,
        const MiniVec<int64_t, 8>* __restrict__ lut_grid_nb2grid_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2cell_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2grid_nb,
        scalar_t* __restrict__ output,
        const int64_t grid_dim,
        const int64_t cells_per_grid,
        const int64_t embedding_dims,
        const int64_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len || !masks[i]) {
        return;
    }

    const functor_t<InterpFunctor> interp_fn;

    int grid_idx = grid_indices[i];
    int cell_idx = cell_indices[i];
    const MiniVec<float, 3>& offset = offsets[i];

    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            lut_grid_nb2grid_idx[grid_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            lut_cell_nb2cell_idx[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            lut_cell_nb2grid_nb[cell_idx];

    // TODO: dispatch feature dims for efficient caching
    auto local_sum_output = MiniVec<scalar_t, 16>::zeros();

    // TODO: revisit strategies for boundary voxels with less than 8 neighbors
    // At current: return directly
    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            return;
        }

        scalar_t weight = 1.0;
        for (int d = 0; d < 3; ++d) {
            int dim_code = (cell_nb >> d) & 1;
            scalar_t x = (dim_code) ? (offset[d]) : (1 - offset[d]);
            weight *= interp_fn(x);
        }

        int cell_nb_idx = neighbor_cell2cell[cell_nb];
        int base_idx =
                (grid_nb_idx * cells_per_grid + cell_nb_idx) * embedding_dims;
        for (int k = 0; k < embedding_dims; ++k) {
            local_sum_output[k] += weight * embeddings[base_idx + k];
        }
    }

    // From local to global memory
    for (int k = 0; k < embedding_dims; ++k) {
        output[i * embedding_dims + k] = local_sum_output[k];
    }
}

at::Tensor query_forward(const at::Tensor& embeddings,
                         const at::Tensor& offsets,
                         const at::Tensor& grid_indices,
                         const at::Tensor& cell_indices,
                         const at::Tensor& masks,
                         const at::Tensor& lut_grid_nb2grid_idx,
                         const at::Tensor& lut_cell_nb2cell_idx,
                         const at::Tensor& lut_cell_nb2grid_nb,
                         const int64_t grid_dim,
                         const std::string& interpolation) {
    // TODO: wise block-thread unrolling
    const int64_t len = grid_indices.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    const int64_t embedding_dims = embeddings.size(2);
    const int64_t num_cells_per_grid = embeddings.size(1);

    at::Tensor output = at::zeros({len, embedding_dims}, embeddings.options());

    AT_DISPATCH_FLOATING_TYPES(embeddings.scalar_type(), "query_forward", [&] {
        DISPATCH_INTERP_FUNCTOR(interpolation, [&] {
            query_forward_kernel<InterpFunctor, scalar_t><<<blocks, threads>>>(
                    embeddings.data_ptr<scalar_t>(),
                    static_cast<MiniVec<float, 3>*>(offsets.data_ptr()),
                    grid_indices.data_ptr<int64_t>(),
                    cell_indices.data_ptr<int64_t>(), masks.data_ptr<bool>(),
                    static_cast<MiniVec<int64_t, 8>*>(
                            lut_grid_nb2grid_idx.data_ptr()),
                    static_cast<MiniVec<int64_t, 8>*>(
                            lut_cell_nb2cell_idx.data_ptr()),
                    static_cast<MiniVec<int64_t, 8>*>(
                            lut_cell_nb2grid_nb.data_ptr()),
                    output.data_ptr<scalar_t>(), grid_dim, num_cells_per_grid,
                    embedding_dims, len);
        });
    });
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

template <typename InterpFunctor, typename DiffInterpFunctor, typename scalar_t>
__global__ void query_backward_forward_kernel(
        const scalar_t* __restrict__ z,
        const scalar_t* __restrict__ embeddings,
        const MiniVec<float, 3>* __restrict__ offsets,
        const int64_t* __restrict__ grid_indices,
        const int64_t* __restrict__ cell_indices,
        const bool* __restrict__ masks,
        const MiniVec<int64_t, 8>* __restrict__ lut_grid_nb2grid_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2cell_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2grid_nb,
        scalar_t* __restrict__ grad_embeddings,
        MiniVec<scalar_t, 3>* __restrict__ grad_offsets,
        const int64_t grid_dim,
        const int64_t cells_per_grid,
        const int64_t embedding_dims,
        const int64_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len || !masks[i]) {
        return;
    }

    const functor_t<InterpFunctor> interp_fn;
    const functor_t<DiffInterpFunctor> diff_interp_fn;

    int grid_idx = grid_indices[i];
    int cell_idx = cell_indices[i];
    const MiniVec<float, 3>& offset = offsets[i];

    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            lut_grid_nb2grid_idx[grid_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            lut_cell_nb2cell_idx[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            lut_cell_nb2grid_nb[cell_idx];

    // Quick neighbor check without polluting output
    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            return;
        }
    }

    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];

        // -1 should never happen
        int grid_nb_idx = neighbor_grid2grid[grid_nb];

        scalar_t weight = 1.0;
        MiniVec<scalar_t, 3> weight_grad = MiniVec<scalar_t, 3>::ones();
        for (int d = 0; d < 3; ++d) {
            int dim_code = (cell_nb >> d) & 1;
            scalar_t x = (dim_code) ? (offset[d]) : (1 - offset[d]);
            scalar_t w = interp_fn(x);
            scalar_t dw = diff_interp_fn(x) * ((dim_code) ? 1 : -1);

            weight *= w;

            weight_grad[0] *= (d == 0) ? dw : w;
            weight_grad[1] *= (d == 1) ? dw : w;
            weight_grad[2] *= (d == 2) ? dw : w;
        }

        int cell_nb_idx = neighbor_cell2cell[cell_nb];
        int base_idx =
                (grid_nb_idx * cells_per_grid + cell_nb_idx) * embedding_dims;

        scalar_t dot = 0.0;
        for (int k = 0; k < embedding_dims; ++k) {
            scalar_t z_k = z[i * embedding_dims + k];
            atomicAdd(&grad_embeddings[base_idx + k], weight * z_k);
            dot += embeddings[base_idx + k] * z_k;
        }
        grad_offsets[i] += dot * weight_grad;
    }
};

std::tuple<at::Tensor, at::Tensor> query_backward_forward(
        const at::Tensor& z,
        const at::Tensor& embeddings,
        const at::Tensor& offsets,
        const at::Tensor& grid_indices,
        const at::Tensor& cell_indices,
        const at::Tensor& masks,
        const at::Tensor& lut_grid_nb2grid_idx,
        const at::Tensor& lut_cell_nb2cell_idx,
        const at::Tensor& lut_cell_nb2grid_nb,
        const int64_t grid_dim,
        const std::string& interpolation) {
    const int64_t len = grid_indices.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    const int64_t embedding_dims = embeddings.size(2);
    const int64_t num_cells_per_grid = embeddings.size(1);
    at::Tensor grad_embeddings = at::zeros_like(embeddings);
    at::Tensor grad_offsets = at::zeros_like(offsets, embeddings.dtype());

    AT_DISPATCH_FLOATING_TYPES(
            embeddings.scalar_type(), "query_backward_forward_kernel", [&] {
                DISPATCH_INTERP_FUNCTOR(interpolation, [&] {
                    query_backward_forward_kernel<InterpFunctor,
                                                  DiffInterpFunctor, scalar_t>
                            <<<blocks, threads>>>(
                                    z.data_ptr<scalar_t>(),
                                    embeddings.data_ptr<scalar_t>(),
                                    static_cast<MiniVec<float, 3>*>(
                                            offsets.data_ptr()),
                                    grid_indices.data_ptr<int64_t>(),
                                    cell_indices.data_ptr<int64_t>(),
                                    masks.data_ptr<bool>(),
                                    static_cast<MiniVec<int64_t, 8>*>(
                                            lut_grid_nb2grid_idx.data_ptr()),
                                    static_cast<MiniVec<int64_t, 8>*>(
                                            lut_cell_nb2cell_idx.data_ptr()),
                                    static_cast<MiniVec<int64_t, 8>*>(
                                            lut_cell_nb2grid_nb.data_ptr()),
                                    grad_embeddings.data_ptr<scalar_t>(),
                                    static_cast<MiniVec<scalar_t, 3>*>(
                                            grad_offsets.data_ptr()),
                                    grid_dim, num_cells_per_grid,
                                    embedding_dims, len);
                });
            });
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    return std::make_tuple(grad_embeddings, grad_offsets);
}

template <typename InterpFunctor, typename DiffInterpFunctor, typename scalar_t>
__global__ void query_backward_backward_kernel(
        const MiniVec<scalar_t, 3>* __restrict__ grad_grad_offset,
        const scalar_t* __restrict__ z,
        const scalar_t* __restrict__ embeddings,
        const MiniVec<float, 3>* __restrict__ offsets,
        const int64_t* __restrict__ grid_indices,
        const int64_t* __restrict__ cell_indices,
        const bool* __restrict__ masks,
        const MiniVec<int64_t, 8>* __restrict__ lut_grid_nb2grid_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2cell_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2grid_nb,
        scalar_t* __restrict__ grad_z,
        scalar_t* __restrict__ grad_embeddings,
        MiniVec<scalar_t, 3>* __restrict__ grad_offsets,
        const int64_t grid_dim,
        const int64_t cells_per_grid,
        const int64_t embedding_dims,
        const int64_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len || !masks[i]) {
        return;
    }

    const functor_t<InterpFunctor> interp_fn;
    const functor_t<DiffInterpFunctor> diff_interp_fn;

    int grid_idx = grid_indices[i];
    int cell_idx = cell_indices[i];
    const MiniVec<float, 3>& offset = offsets[i];

    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            lut_grid_nb2grid_idx[grid_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            lut_cell_nb2cell_idx[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            lut_cell_nb2grid_nb[cell_idx];

    // Quick neighbor check
    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            return;
        }
    }

    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];

        // Should never be -1
        int grid_nb_idx = neighbor_grid2grid[grid_nb];

        MiniVec<scalar_t, 3> weight_grad = MiniVec<scalar_t, 3>::ones();
        for (int d = 0; d < 3; ++d) {
            int dim_code = (cell_nb >> d) & 1;
            scalar_t x = (dim_code) ? (offset[d]) : (1 - offset[d]);
            scalar_t w = interp_fn(x);
            scalar_t dw = diff_interp_fn(x) * ((dim_code) ? 1 : -1);

            weight_grad[0] *= (d == 0) ? dw : w;
            weight_grad[1] *= (d == 1) ? dw : w;
            weight_grad[2] *= (d == 2) ? dw : w;
        }
        int cell_nb_idx = neighbor_cell2cell[cell_nb];
        int base_idx_lhs =
                (grid_nb_idx * cells_per_grid + cell_nb_idx) * embedding_dims;
        int base_idx_rhs = i * embedding_dims;

        scalar_t dot = weight_grad.dot(grad_grad_offset[i]);
        for (int k = 0; k < embedding_dims; ++k) {
            atomicAdd(&grad_embeddings[base_idx_lhs + k],
                      dot * z[base_idx_rhs + k]);

            // TODO: put it in a local feature as well
            grad_z[base_idx_rhs + k] += dot * embeddings[base_idx_lhs + k];
        }
    }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> query_backward_backward(
        const at::Tensor& grad_grad_embeddings,
        const at::Tensor& grad_grad_offset,
        const at::Tensor& z,
        const at::Tensor& embeddings,
        const at::Tensor& offsets,
        // queried x via a non-differentiable hash map lookup beforehand
        const at::Tensor& grid_indices,
        const at::Tensor& cell_indices,
        const at::Tensor& masks,
        // sparse luts
        const at::Tensor& lut_grid_nb2grid_idx,  // (N, 1)
        // dense luts
        const at::Tensor& lut_cell_nb2cell_idx,  // (M^3, 8)
        const at::Tensor& lut_cell_nb2grid_nb,   // (M^3, 8)
        const int64_t grid_dim,
        const std::string& interpolation) {
    const int64_t len = grid_indices.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    const int64_t embedding_dims = embeddings.size(2);
    const int64_t num_cells_per_grid = embeddings.size(1);

    at::Tensor grad_z = at::zeros_like(z);
    at::Tensor grad_embeddings = at::zeros_like(embeddings);

    // ignored for now
    at::Tensor grad_offsets = at::zeros_like(offsets, embeddings.dtype());

    AT_DISPATCH_FLOATING_TYPES(
            embeddings.scalar_type(), "query_backward_backward", [&] {
                DISPATCH_INTERP_FUNCTOR(interpolation, [&] {
                    query_backward_backward_kernel<
                            InterpFunctor, DiffInterpFunctor,
                            scalar_t><<<blocks, threads>>>(
                            // unused grad_grad_embeddings.data_ptr<scalar_t>(),
                            static_cast<MiniVec<scalar_t, 3>*>(
                                    grad_grad_offset.data_ptr()),
                            z.data_ptr<scalar_t>(),
                            embeddings.data_ptr<scalar_t>(),
                            static_cast<MiniVec<float, 3>*>(offsets.data_ptr()),
                            grid_indices.data_ptr<int64_t>(),
                            cell_indices.data_ptr<int64_t>(),
                            masks.data_ptr<bool>(),
                            static_cast<MiniVec<int64_t, 8>*>(
                                    lut_grid_nb2grid_idx.data_ptr()),
                            static_cast<MiniVec<int64_t, 8>*>(
                                    lut_cell_nb2cell_idx.data_ptr()),
                            static_cast<MiniVec<int64_t, 8>*>(
                                    lut_cell_nb2grid_nb.data_ptr()),
                            grad_z.data_ptr<scalar_t>(),
                            grad_embeddings.data_ptr<scalar_t>(),
                            static_cast<MiniVec<scalar_t, 3>*>(
                                    grad_offsets.data_ptr()),
                            grid_dim, num_cells_per_grid, embedding_dims, len);
                });
            });
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_tuple(grad_z, grad_embeddings, grad_offsets);
}

template <typename scalar_t>
__global__ void isosurface_extraction_kernel(
        const scalar_t* __restrict__ sdfs,
        const scalar_t* __restrict__ weights,
        const int64_t* __restrict__ grid_indices,
        const MiniVec<int, 3>* __restrict__ grid_coords_table,
        const MiniVec<int64_t, 8>* __restrict__ lut_grid_nb2grid_idx,
        const MiniVec<int, 3>* __restrict__ cell_coords_table,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2cell_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2grid_nb,
        int* __restrict__ output_counter,
        MiniVec<float, 3>* __restrict__ output_positions,
        const float iso_value,
        const float weight_thr,
        const int grid_dim,
        const int num_cells_per_grid,
        const int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }

    int sparse_i = i / num_cells_per_grid;
    int dense_i = i % num_cells_per_grid;

    int grid_idx = grid_indices[sparse_i];
    const MiniVec<int, 3>& sparse_coord = grid_coords_table[grid_idx];
    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            lut_grid_nb2grid_idx[grid_idx];

    int cell_idx = dense_i;
    const MiniVec<int, 3>& cell_coord = cell_coords_table[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            lut_cell_nb2cell_idx[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            lut_cell_nb2grid_nb[cell_idx];

    scalar_t self_sdf = sdfs[grid_idx * num_cells_per_grid + cell_idx];
    scalar_t self_weight = weights[grid_idx * num_cells_per_grid + cell_idx];
    scalar_t self_isodiff = self_sdf - iso_value;

    if (self_weight < weight_thr) {
        return;
    }

    MiniVec<float, 3> offset = MiniVec<float, 3>::zeros();
    for (int d = 0; d < 3; ++d) {
        // Look for 3 neighbors in each dimension
        offset[d] = 1;

        int cell_nb = 1 << d;
        int grid_nb = neighbor_cell2grid[cell_nb];

        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            return;
        }
        int cell_nb_idx = neighbor_cell2cell[cell_nb];

        scalar_t nb_sdf = sdfs[grid_nb_idx * num_cells_per_grid + cell_nb_idx];
        scalar_t nb_weight =
                weights[grid_nb_idx * num_cells_per_grid + cell_nb_idx];
        scalar_t nb_isodiff = nb_sdf - iso_value;

        if (self_isodiff * nb_isodiff < 0 && nb_weight >= weight_thr) {
            int output_idx = atomicAdd(output_counter, 1);

            if (output_positions) {
                float ratio = self_isodiff / (self_isodiff - nb_isodiff);
                MiniVec<int, 3> self_position =
                        sparse_coord * grid_dim + cell_coord;
                MiniVec<float, 3> surface_position =
                        self_position.template cast<float>() + ratio * offset;
                output_positions[output_idx] = surface_position;
            }
        }

        offset[d] = 0;
    }
}

at::Tensor isosurface_extraction(
        const at::Tensor& sdfs,     // (num_embeddings, dense_res^3, 1)
        const at::Tensor& weights,  // (num_embeddings, dense_res^3, 1)
        const at::Tensor& grid_indices,
        const at::Tensor& grid_coords_table,
        const at::Tensor& lut_grid_nb2grid_idx,
        const at::Tensor& cell_coords_table,
        const at::Tensor& lut_cell_nb2cell_idx,  // (M^3, 8)
        const at::Tensor& lut_cell_nb2grid_nb,   // (M^3, 8)
        const int64_t grid_dim,
        const float iso_value,
        const float weight_thr) {
    // TODO: wise block-thread unrolling
    const int64_t len = grid_indices.size(0) * cell_coords_table.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    at::Tensor output_counter = at::zeros({}, grid_coords_table.options());

    int num_cells_per_grid = sdfs.size(1);

    AT_DISPATCH_FLOATING_TYPES(
            sdfs.scalar_type(), "isosurface_extraction_kernel", ([&] {
                isosurface_extraction_kernel<scalar_t><<<blocks, threads>>>(
                        sdfs.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
                        grid_indices.data_ptr<int64_t>(),
                        static_cast<MiniVec<int, 3>*>(
                                grid_coords_table.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                lut_grid_nb2grid_idx.data_ptr()),
                        static_cast<MiniVec<int, 3>*>(
                                cell_coords_table.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                lut_cell_nb2cell_idx.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                lut_cell_nb2grid_nb.data_ptr()),
                        output_counter.data_ptr<int>(), nullptr, iso_value,
                        weight_thr, grid_dim, num_cells_per_grid, len);
            }));
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    at::Tensor output_positions =
            at::zeros({output_counter.item<int>(), 3}, sdfs.options());
    output_counter = at::zeros({}, grid_coords_table.options());

    AT_DISPATCH_FLOATING_TYPES(
            sdfs.scalar_type(), "isosurface_extraction_kernel", ([&] {
                isosurface_extraction_kernel<scalar_t><<<blocks, threads>>>(
                        sdfs.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
                        grid_indices.data_ptr<int64_t>(),
                        static_cast<MiniVec<int, 3>*>(
                                grid_coords_table.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                lut_grid_nb2grid_idx.data_ptr()),
                        static_cast<MiniVec<int, 3>*>(
                                cell_coords_table.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                lut_cell_nb2cell_idx.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                lut_cell_nb2grid_nb.data_ptr()),
                        output_counter.data_ptr<int>(),
                        static_cast<MiniVec<float, 3>*>(
                                output_positions.data_ptr()),
                        iso_value, weight_thr, grid_dim, num_cells_per_grid,
                        len);
            }));
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return output_positions;
}

template <typename scalar_t>
__global__ void marching_cubes_table_idx_kernel(
        const scalar_t* __restrict__ sdfs,
        const scalar_t* __restrict__ weights,
        const int64_t* __restrict__ grid_indices,
        const MiniVec<int64_t, 8>* __restrict__ lut_grid_nb2grid_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2cell_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2grid_nb,
        int* __restrict__ table_indices,
        MiniVec<int, 3>* __restrict__ edge_vertex_indices,  // 3 per cell in the
                                                            // sparse-dense grid
        const float iso_value,
        const float weight_thr,
        const int grid_dim,
        const int num_cells_per_grid,
        const int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    int sparse_i = i / num_cells_per_grid;
    int dense_i = i % num_cells_per_grid;

    int grid_idx = grid_indices[sparse_i];
    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            lut_grid_nb2grid_idx[grid_idx];

    int cell_idx = dense_i;
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            lut_cell_nb2cell_idx[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            lut_cell_nb2grid_nb[cell_idx];

    scalar_t self_sdf = sdfs[grid_idx * num_cells_per_grid + cell_idx];
    scalar_t self_weight = weights[grid_idx * num_cells_per_grid + cell_idx];
    scalar_t self_isodiff = self_sdf - iso_value;

    if (self_weight < weight_thr) {
        return;
    }

    int table_idx = 0;
    for (int corner = 0; corner < 8; ++corner) {
        int cell_nb = nb_mc_vtx_map[corner];
        int grid_nb = neighbor_cell2grid[cell_nb];

        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            return;
        }
        int cell_nb_idx = neighbor_cell2cell[cell_nb];

        scalar_t nb_sdf = sdfs[grid_nb_idx * num_cells_per_grid + cell_nb_idx];
        scalar_t nb_weight =
                weights[grid_nb_idx * num_cells_per_grid + cell_nb_idx];
        scalar_t nb_isodiff = nb_sdf - iso_value;

        if (nb_weight < weight_thr) {
            return;
        }

        table_idx |= ((nb_isodiff < 0) << corner);
    }

    // No vertex, return
    if (table_idx == 0 || table_idx == 255) {
        return;
    }
    table_indices[grid_idx * num_cells_per_grid + cell_idx] = table_idx;

    // 1st 12 bits encodes edge-isosurface intersection
    int edges_encoding = edge_table[table_idx];
    for (int edge = 0; edge < 12; ++edge) {
        if (!(edges_encoding & (1 << edge))) {
            continue;
        }

        // A cell is in charge of 3 edges at xyz directions
        // First decide the cell in charge of this edge (12 edges for a cell,
        // could be managed by a neighbor)
        // and the edge direction
        int cell_nb = nb_mc_vtx_map[edge_to_vert[edge][0]];  // [0, 8)
        int edge_dir = edge_shifts[edge][3];                 // [0, 1, 2]

        int grid_nb = neighbor_cell2grid[cell_nb];

        // must be valid from the table decision
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        int cell_nb_idx = neighbor_cell2cell[cell_nb];

        // Placeholder for the vertex idx
        edge_vertex_indices[grid_nb_idx * num_cells_per_grid + cell_nb_idx]
                           [edge_dir] = -1;
    }
}

template <typename scalar_t>
__global__ void marching_cubes_vertex_kernel(
        const scalar_t* __restrict__ sdfs,
        const scalar_t* __restrict__ weights,
        const int64_t* __restrict__ grid_indices,
        const MiniVec<int, 3>* __restrict__ grid_coords_table,
        const MiniVec<int64_t, 8>* __restrict__ lut_grid_nb2grid_idx,
        const MiniVec<int, 3>* __restrict__ cell_coords_table,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2cell_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2grid_nb,
        MiniVec<int, 3>* __restrict__ edge_vertex_indices,  // 3 per cell in the
                                                            // sparse-dense grid
        int* __restrict__ vertex_counter,
        MiniVec<float, 3>* __restrict__ vertex_positions,
        const float iso_value,
        const float weight_thr,
        const int grid_dim,
        const int num_cells_per_grid,
        const int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    int sparse_i = i / num_cells_per_grid;
    int dense_i = i % num_cells_per_grid;

    int grid_idx = grid_indices[sparse_i];
    const MiniVec<int, 3>& sparse_coord = grid_coords_table[grid_idx];
    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            lut_grid_nb2grid_idx[grid_idx];

    int cell_idx = dense_i;
    const MiniVec<int, 3>& cell_coord = cell_coords_table[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            lut_cell_nb2cell_idx[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            lut_cell_nb2grid_nb[cell_idx];

    scalar_t self_sdf = sdfs[grid_idx * num_cells_per_grid + cell_idx];
    scalar_t self_weight = weights[grid_idx * num_cells_per_grid + cell_idx];
    scalar_t self_isodiff = self_sdf - iso_value;

    if (self_weight < weight_thr) {
        return;
    }

    MiniVec<float, 3> offset = MiniVec<float, 3>::zeros();
    for (int d = 0; d < 3; ++d) {
        if (edge_vertex_indices[grid_idx * num_cells_per_grid + cell_idx][d] !=
            -1) {
            continue;
        }
        offset[d] = 1;

        int cell_nb = 1 << d;
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        int cell_nb_idx = neighbor_cell2cell[cell_nb];

        scalar_t nb_sdf = sdfs[grid_nb_idx * num_cells_per_grid + cell_nb_idx];
        scalar_t nb_isodiff = nb_sdf - iso_value;

        int output_idx = atomicAdd(vertex_counter, 1);

        float ratio = self_isodiff / (self_isodiff - nb_isodiff);
        MiniVec<int, 3> self_position = sparse_coord * grid_dim + cell_coord;
        MiniVec<float, 3> surface_position =
                self_position.template cast<float>() + ratio * offset;
        edge_vertex_indices[grid_idx * num_cells_per_grid + cell_idx][d] =
                output_idx;
        offset[d] = 0;
        vertex_positions[output_idx] = surface_position;
    }
}

template <typename scalar_t>
__global__ void marching_cubes_triangle_kernel(
        scalar_t* __restrict__ sdfs,
        const scalar_t* __restrict__ weights,
        const int64_t* __restrict__ grid_indices,
        const MiniVec<int64_t, 8>* __restrict__ lut_grid_nb2grid_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2cell_idx,
        const MiniVec<int64_t, 8>* __restrict__ lut_cell_nb2grid_nb,
        const int* __restrict__ table_indices,
        const MiniVec<int, 3>* __restrict__ edge_vertex_indices,
        int* __restrict__ triangle_counter,
        MiniVec<int, 3>* __restrict__ triangle_indices,
        const float iso_value,
        const float weight_thr,
        const int grid_dim,
        const int num_cells_per_grid,
        const int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    int sparse_i = i / num_cells_per_grid;
    int dense_i = i % num_cells_per_grid;

    int grid_idx = grid_indices[sparse_i];
    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            lut_grid_nb2grid_idx[grid_idx];

    int cell_idx = dense_i;
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            lut_cell_nb2cell_idx[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            lut_cell_nb2grid_nb[cell_idx];

    int table_idx = table_indices[grid_idx * num_cells_per_grid + cell_idx];
    if (table_idx == 0) {
        return;
    }
    for (int tri = 0; tri < 16; tri += 3) {
        if (tri_table[table_idx][tri] == -1) {
            return;
        }
        int tri_idx = atomicAdd(triangle_counter, 1);

        if (triangle_indices == nullptr) continue;

        for (int vertex = 0; vertex < 3; ++vertex) {
            int edge = tri_table[table_idx][tri + vertex];

            int cell_nb = nb_mc_vtx_map[edge_to_vert[edge][0]];

            // extract the vertex idx from the edge
            int grid_nb = neighbor_cell2grid[cell_nb];
            int edge_dir = edge_shifts[edge][3];
            int grid_nb_idx = neighbor_grid2grid[grid_nb];
            int cell_nb_idx = neighbor_cell2cell[cell_nb];

            int vertex_idx =
                    edge_vertex_indices[grid_nb_idx * num_cells_per_grid +
                                        cell_nb_idx][edge_dir];

            triangle_indices[tri_idx][(2 - vertex)] = vertex_idx;
        }
    }
}

std::tuple<at::Tensor, at::Tensor> marching_cubes(
        const at::Tensor& sdfs,     // (num_embeddings, dense_res^3, 1)
        const at::Tensor& weights,  // (num_embeddings, dense_res^3, 1)

        const at::Tensor& grid_indices,          // (N, 1)
        const at::Tensor& grid_coords_table,     // (N, 3)
        const at::Tensor& lut_grid_nb2grid_idx,  // (N, 8) [-1 for
                                                 // invalid neighbors]
        const at::Tensor& cell_coords_table,     // (dense_res^3, 1)
        const at::Tensor& lut_cell_nb2cell_idx,  // (dense_res^3, 8)
        const at::Tensor& lut_cell_nb2grid_nb,   // (dense_res^3, 8)
        const int64_t grid_dim,
        const float iso_value,
        const float weight_thr) {
    const int64_t num_embeddings = sdfs.size(0);
    const int64_t num_cells_per_grid = sdfs.size(1);

    const int64_t num_sparse_grids = grid_indices.size(0);
    const int64_t len = num_sparse_grids * num_cells_per_grid;

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    // Determine marching cubes table idx and vertex existence per cell
    auto options =
            at::TensorOptions().dtype(torch::kInt32).device(sdfs.device());

    at::Tensor table_indices =
            at::zeros({num_embeddings, num_cells_per_grid}, options);
    at::Tensor edge_vertex_indices =
            at::zeros({num_embeddings, num_cells_per_grid, 3}, options);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    at::Tensor triangles = at::zeros({0, 3}, options);
    at::Tensor vertex_positions = at::zeros({0, 3}, sdfs.options());

    AT_DISPATCH_FLOATING_TYPES(sdfs.scalar_type(), "marching_cubes", [&] {
        marching_cubes_table_idx_kernel<scalar_t><<<blocks, threads>>>(
                sdfs.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
                grid_indices.data_ptr<int64_t>(),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_grid_nb2grid_idx.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_cell_nb2cell_idx.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_cell_nb2grid_nb.data_ptr()),
                table_indices.data_ptr<int>(),
                static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
                iso_value, weight_thr, grid_dim, num_cells_per_grid, len);
        C10_CUDA_CHECK(cudaDeviceSynchronize());

        int num_vertices = edge_vertex_indices.eq(-1).sum().item<int>();
        at::Tensor vertex_counter = at::zeros({}, options);
        vertex_positions = at::zeros({num_vertices, 3}, sdfs.options());
        marching_cubes_vertex_kernel<scalar_t><<<blocks, threads>>>(
                sdfs.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
                grid_indices.data_ptr<int64_t>(),
                static_cast<MiniVec<int, 3>*>(grid_coords_table.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_grid_nb2grid_idx.data_ptr()),
                static_cast<MiniVec<int, 3>*>(cell_coords_table.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_cell_nb2cell_idx.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_cell_nb2grid_nb.data_ptr()),
                static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
                vertex_counter.data_ptr<int>(),
                static_cast<MiniVec<float, 3>*>(vertex_positions.data_ptr()),
                iso_value, weight_thr, grid_dim, num_cells_per_grid, len);
        C10_CUDA_CHECK(cudaDeviceSynchronize());

        at::Tensor triangle_counter = at::zeros({}, options);
        marching_cubes_triangle_kernel<scalar_t><<<blocks, threads>>>(
                sdfs.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
                grid_indices.data_ptr<int64_t>(),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_grid_nb2grid_idx.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_cell_nb2cell_idx.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_cell_nb2grid_nb.data_ptr()),
                table_indices.data_ptr<int>(),
                static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
                triangle_counter.data_ptr<int>(), nullptr, iso_value,
                weight_thr, grid_dim, num_cells_per_grid, len);
        C10_CUDA_CHECK(cudaDeviceSynchronize());

        triangles = at::zeros({triangle_counter.item<int>(), 3}, options);
        triangle_counter.zero_();
        marching_cubes_triangle_kernel<scalar_t><<<blocks, threads>>>(
                sdfs.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
                grid_indices.data_ptr<int64_t>(),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_grid_nb2grid_idx.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_cell_nb2cell_idx.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        lut_cell_nb2grid_nb.data_ptr()),
                table_indices.data_ptr<int>(),
                static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
                triangle_counter.data_ptr<int>(),
                static_cast<MiniVec<int, 3>*>(triangles.data_ptr()), iso_value,
                weight_thr, grid_dim, num_cells_per_grid, len);
        C10_CUDA_CHECK(cudaDeviceSynchronize());
    });
    return {triangles, vertex_positions};
}

// Only provide forward convolution (interpolation) for now
// TODO: add backward convolution and accelerate
template <typename scalar_t>
__global__ void convolution_forward_kernel(
        const scalar_t* __restrict__ inputs,
        const scalar_t* __restrict__ weights,
        const bool* __restrict__ masks,
        const int64_t* __restrict__ grid_indices,
        const int64_t* __restrict__ cell_indices,
        const int64_t* __restrict__ lut_grid_nb2grid_idx,
        const int64_t* __restrict__ lut_cell_nb2cell_idx,
        const int64_t* __restrict__ lut_cell_nb2grid_nb,
        scalar_t* outputs,
        const int64_t num_cell_nbs,
        const int64_t num_grid_nbs,
        const int64_t grid_dim,
        const int64_t num_cells_per_grid,
        const int64_t embedding_dims,
        const int64_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len || !masks[i]) return;

    int grid_i = i / num_cells_per_grid;
    int cell_i = i % num_cells_per_grid;

    int grid_idx = grid_indices[grid_i];
    int cell_idx = cell_indices[cell_i];

    const int64_t* grid_nb2grid_idx =
            lut_grid_nb2grid_idx + num_grid_nbs * grid_idx;
    const int64_t* cell_nb2cell_idx =
            lut_cell_nb2cell_idx + num_cell_nbs * cell_idx;
    const int64_t* cell_nb2grid_nb =
            lut_cell_nb2grid_nb + num_cell_nbs * cell_idx;

    auto local_sum_output = MiniVec<scalar_t, 16>::zeros();
    for (int k = 0; k < num_cell_nbs; ++k) {
        int grid_nb = cell_nb2grid_nb[k];
        int grid_nb_idx = grid_nb2grid_idx[grid_nb];
        if (grid_nb_idx < 0) continue;

        int cell_nb_idx = cell_nb2cell_idx[k];
        const scalar_t* input =
                inputs + (grid_nb_idx * num_cells_per_grid + cell_nb_idx) *
                                 embedding_dims;
        const scalar_t* weight = weights + k * embedding_dims;
        for (int c = 0; c < embedding_dims; ++c) {
            // Simplified; general case it would be a matmul
            local_sum_output[c] += input[c] * weight[c];
        }
    }
    auto output = outputs +
                  (grid_idx * num_cells_per_grid + cell_idx) * embedding_dims;
    for (int c = 0; c < embedding_dims; ++c) {
        output[c] = local_sum_output[c];
    }
}

at::Tensor convolution_forward(
        const at::Tensor& inputs,
        const at::Tensor& weights,  // (K window)
        const at::Tensor& masks,
        const at::Tensor& grid_indices,          // (N, 1)
        const at::Tensor& cell_indices,          // (M^3, 8)
        const at::Tensor& lut_grid_nb2grid_idx,  // (N, K^3)
        const at::Tensor& lut_cell_nb2cell_idx,  // (M^3, K^3)
        const at::Tensor& lut_cell_nb2grid_nb,   // (M^3, K^3)
        const int64_t grid_dim) {
    at::Tensor outputs = at::zeros_like(inputs);
    const int64_t embedding_dims = inputs.size(2);

    const int64_t num_grid_nbs = lut_grid_nb2grid_idx.size(1);
    const int64_t num_cell_nbs = lut_cell_nb2cell_idx.size(1);
    const int64_t num_cells_per_grid = inputs.size(1);
    const int64_t num_grids = grid_indices.size(0);
    const int64_t len = num_grids * num_cells_per_grid;

    const int threads = 256;
    const int blocks = (len + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(
            inputs.scalar_type(), "convolution_forward", [&] {
                convolution_forward_kernel<scalar_t><<<blocks, threads>>>(
                        inputs.data_ptr<scalar_t>(),
                        weights.data_ptr<scalar_t>(), masks.data_ptr<bool>(),
                        grid_indices.data_ptr<int64_t>(),
                        cell_indices.data_ptr<int64_t>(),
                        lut_grid_nb2grid_idx.data_ptr<int64_t>(),
                        lut_cell_nb2cell_idx.data_ptr<int64_t>(),
                        lut_cell_nb2grid_nb.data_ptr<int64_t>(),
                        outputs.data_ptr<scalar_t>(), num_cell_nbs,
                        num_grid_nbs, grid_dim, num_cells_per_grid,
                        embedding_dims, len);
            });
    C10_CUDA_CHECK(cudaGetLastError());
    return outputs;
}
