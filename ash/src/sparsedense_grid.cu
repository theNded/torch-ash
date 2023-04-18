#include <c10/cuda/CUDAException.h>

#include "mc_macros.h"
#include "minivec.h"
#include "sparsedense_grid.h"

const float eps = 1e-6;
// Now only dispatch dtypes, all the queries are for 3D
// TODO: dispatch for 2D/4D later
template <typename scalar_t>
__global__ void query_forward_kernel(
        const scalar_t* __restrict__ embeddings,
        const MiniVec<float, 3>* __restrict__ offsets,
        const int64_t* __restrict__ grid_indices,
        const int64_t* __restrict__ cell_indices,
        const bool* __restrict__ masks,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_grid2grid,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2cell,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2grid,
        scalar_t* __restrict__ output,
        const int64_t grid_dim,
        const int64_t cells_per_grid,
        const int64_t embedding_dims,
        const int64_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len || !masks[i]) {
        return;
    }

    int grid_idx = grid_indices[i];
    int cell_idx = cell_indices[i];
    const MiniVec<float, 3>& offset = offsets[i];

    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            neighbor_table_grid2grid[grid_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            neighbor_table_cell2cell[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            neighbor_table_cell2grid[cell_idx];

    // TODO: dispatch or better serialization
    MiniVec<scalar_t, 16> sum_output = MiniVec<scalar_t, 16>::zeros();
    float sum_weight = 0.0;
    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            continue;
        }

        float weight = 1.0;
        for (int d = 0; d < 3; ++d) {
            int dim_code = (cell_nb >> d) & 1;
            weight *= (dim_code) ? (offset[d]) : (1 - offset[d]);
        }

        int cell_nb_idx = neighbor_cell2cell[cell_nb];
        int base_idx =
                (grid_nb_idx * cells_per_grid + cell_nb_idx) * embedding_dims;
        for (int k = 0; k < embedding_dims; ++k) {
            sum_output[k] += weight * embeddings[base_idx + k];
        }
        sum_weight += weight;
    }

    if (sum_weight < eps) {
        return;
    }

    for (int k = 0; k < embedding_dims; ++k) {
        output[i * embedding_dims + k] = sum_output[k] / sum_weight;
    }
}

at::Tensor query_forward(
        const at::Tensor&
                embeddings,  // (num_embeddings, dense_res^3, embedding_dims)
        const at::Tensor& offsets,
        const at::Tensor& grid_indices,
        const at::Tensor& cell_indices,
        const at::Tensor& masks,
        const at::Tensor& neighbor_table_grid2grid,
        const at::Tensor& neighbor_table_cell2cell,
        const at::Tensor& neighbor_table_cell2grid,
        const int64_t grid_dim) {
    // TODO: wise block-thread unrolling
    const int64_t len = grid_indices.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    const int64_t embedding_dims = embeddings.size(2);
    const int64_t num_cells_per_grid = embeddings.size(1);

    at::Tensor output = at::zeros({len, embedding_dims}, embeddings.options());

    AT_DISPATCH_FLOATING_TYPES(embeddings.scalar_type(), "query_forward", [&] {
        query_forward_kernel<scalar_t><<<blocks, threads>>>(
                embeddings.data_ptr<scalar_t>(),
                static_cast<MiniVec<float, 3>*>(offsets.data_ptr()),
                grid_indices.data_ptr<int64_t>(),
                cell_indices.data_ptr<int64_t>(), masks.data_ptr<bool>(),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_grid2grid.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2cell.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2grid.data_ptr()),
                output.data_ptr<scalar_t>(), grid_dim, num_cells_per_grid,
                embedding_dims, len);
    });
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

template <typename scalar_t>
__global__ void query_backward_forward_kernel(
        const scalar_t* __restrict__ z,
        const scalar_t* __restrict__ embeddings,
        const MiniVec<float, 3>* __restrict__ offsets,
        const int64_t* __restrict__ grid_indices,
        const int64_t* __restrict__ cell_indices,
        const bool* __restrict__ masks,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_grid2grid,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2cell,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2grid,
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

    int grid_idx = grid_indices[i];
    int cell_idx = cell_indices[i];
    const MiniVec<float, 3>& offset = offsets[i];

    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            neighbor_table_grid2grid[grid_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            neighbor_table_cell2cell[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            neighbor_table_cell2grid[cell_idx];

    scalar_t sum_weight = 0.0;
    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            continue;
        }

        scalar_t weight = 1.0;
        for (int d = 0; d < 3; ++d) {
            int dim_code = (cell_nb >> d) & 1;
            scalar_t w = (dim_code) ? (offset[d]) : (1 - offset[d]);
            weight *= w;
        }
        sum_weight += weight;
    }
    if (sum_weight < eps) {
        return;
    }

    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            continue;
        }

        scalar_t weight = 1.0;
        MiniVec<scalar_t, 3> weight_grad = MiniVec<scalar_t, 3>::ones();
        for (int d = 0; d < 3; ++d) {
            int dim_code = (cell_nb >> d) & 1;
            float w = (dim_code) ? (offset[d]) : (1 - offset[d]);
            float dw = (dim_code) ? (1) : (-1);
            weight *= w;

            weight_grad[0] *= (d == 0) ? dw : w;
            weight_grad[1] *= (d == 1) ? dw : w;
            weight_grad[2] *= (d == 2) ? dw : w;
        }

        int cell_nb_idx = neighbor_cell2cell[cell_nb];
        int base_idx =
                (grid_nb_idx * cells_per_grid + cell_nb_idx) * embedding_dims;

        scalar_t dot = 0.0;
        scalar_t normalized_weight = weight / sum_weight;
        for (int k = 0; k < embedding_dims; ++k) {
            atomicAdd(&grad_embeddings[base_idx + k],
                      normalized_weight * z[i * embedding_dims + k]);
            dot += embeddings[base_idx + k] * z[i * embedding_dims + k];
        }
        grad_offsets[i] += (dot / sum_weight) * weight_grad;
    }
};

std::tuple<at::Tensor, at::Tensor> query_backward_forward(
        const at::Tensor& z,
        const at::Tensor& embeddings,
        const at::Tensor& offsets,
        const at::Tensor& grid_indices,
        const at::Tensor& cell_indices,
        const at::Tensor& masks,
        const at::Tensor& neighbor_table_grid2grid,
        const at::Tensor& neighbor_table_cell2cell,
        const at::Tensor& neighbor_table_cell2grid,
        const int64_t grid_dim) {
    const int64_t len = grid_indices.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    const int64_t embedding_dims = embeddings.size(2);
    const int64_t num_cells_per_grid = embeddings.size(1);

    at::Tensor grad_embeddings = at::zeros_like(embeddings);
    at::Tensor grad_offsets = at::zeros_like(offsets, embeddings.dtype());

    AT_DISPATCH_FLOATING_TYPES(
            embeddings.scalar_type(), "query_backward_forward_kernel", [&] {
                query_backward_forward_kernel<scalar_t><<<blocks, threads>>>(
                        z.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(),
                        static_cast<MiniVec<float, 3>*>(offsets.data_ptr()),
                        grid_indices.data_ptr<int64_t>(),
                        cell_indices.data_ptr<int64_t>(),
                        masks.data_ptr<bool>(),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_grid2grid.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_cell2cell.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_cell2grid.data_ptr()),
                        grad_embeddings.data_ptr<scalar_t>(),
                        static_cast<MiniVec<scalar_t, 3>*>(
                                grad_offsets.data_ptr()),
                        grid_dim, num_cells_per_grid, embedding_dims, len);
            });
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_tuple(grad_embeddings, grad_offsets);
}

template <typename scalar_t>
__global__ void query_backward_backward_kernel(
        const MiniVec<scalar_t, 3>* __restrict__ grad_grad_offset,
        const scalar_t* __restrict__ z,
        const MiniVec<float, 3>* __restrict__ offsets,
        const int64_t* __restrict__ grid_indices,
        const int64_t* __restrict__ cell_indices,
        const bool* __restrict__ masks,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_grid2grid,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2cell,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2grid,
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

    int grid_idx = grid_indices[i];
    int cell_idx = cell_indices[i];
    const MiniVec<float, 3>& offset = offsets[i];

    const MiniVec<int64_t, 8>& neighbor_grid2grid =
            neighbor_table_grid2grid[grid_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            neighbor_table_cell2cell[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            neighbor_table_cell2grid[cell_idx];

    scalar_t sum_weight = 0.0;
    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            continue;
        }

        scalar_t weight = 1.0;
        for (int d = 0; d < 3; ++d) {
            int dim_code = (cell_nb >> d) & 1;
            scalar_t w = (dim_code) ? (offset[d]) : (1 - offset[d]);
            weight *= w;
        }
        sum_weight += weight;
    }
    if (sum_weight < eps) {
        return;
    }

    for (int cell_nb = 0; cell_nb < 8; ++cell_nb) {
        int grid_nb = neighbor_cell2grid[cell_nb];
        int grid_nb_idx = neighbor_grid2grid[grid_nb];
        if (grid_nb_idx == -1) {
            continue;
        }

        MiniVec<scalar_t, 3> weight_grad = MiniVec<scalar_t, 3>::ones();
        for (int d = 0; d < 3; ++d) {
            int dim_code = (cell_nb >> d) & 1;
            scalar_t w = (dim_code) ? (offset[d]) : (1 - offset[d]);
            scalar_t dw = (dim_code) ? (1) : (-1);

            weight_grad[0] *= (d == 0) ? dw : w;
            weight_grad[1] *= (d == 1) ? dw : w;
            weight_grad[2] *= (d == 2) ? dw : w;
        }
        int cell_nb_idx = neighbor_cell2cell[cell_nb];
        int base_idx_lhs =
                (grid_nb_idx * cells_per_grid + cell_nb_idx) * embedding_dims;
        int base_idx_rhs = i * embedding_dims;

        scalar_t dot = weight_grad.dot(grad_grad_offset[i]);
        scalar_t factor = dot / sum_weight;
        for (int k = 0; k < embedding_dims; ++k) {
            atomicAdd(&grad_embeddings[base_idx_lhs + k],
                      factor * z[base_idx_rhs + k]);
        }
    }
};

std::tuple<at::Tensor, at::Tensor> query_backward_backward(
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
        const at::Tensor& neighbor_table_grid2grid,  // (N, 1)
        // dense luts
        const at::Tensor& neighbor_table_cell2cell,  // (M^3, 8)
        const at::Tensor& neighbor_table_cell2grid,  // (M^3, 8)
        const int64_t grid_dim) {
    const int64_t len = grid_indices.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    const int64_t embedding_dims = embeddings.size(2);
    const int64_t num_cells_per_grid = embeddings.size(1);

    at::Tensor grad_embeddings = at::zeros_like(embeddings);

    // ignored for now
    at::Tensor grad_offsets = at::zeros_like(offsets, embeddings.dtype());

    AT_DISPATCH_FLOATING_TYPES(
            embeddings.scalar_type(), "query_backward_backward", [&] {
                query_backward_backward_kernel<scalar_t><<<blocks, threads>>>(
                        // unused grad_grad_embeddings.data_ptr<scalar_t>(),
                        static_cast<MiniVec<scalar_t, 3>*>(
                                grad_grad_offset.data_ptr()),
                        z.data_ptr<scalar_t>(),
                        static_cast<MiniVec<float, 3>*>(offsets.data_ptr()),
                        grid_indices.data_ptr<int64_t>(),
                        cell_indices.data_ptr<int64_t>(),
                        masks.data_ptr<bool>(),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_grid2grid.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_cell2cell.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_cell2grid.data_ptr()),
                        grad_embeddings.data_ptr<scalar_t>(),
                        static_cast<MiniVec<scalar_t, 3>*>(
                                grad_offsets.data_ptr()),
                        grid_dim, num_cells_per_grid, embedding_dims, len);
            });
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_tuple(grad_embeddings, grad_offsets);
}

template <typename scalar_t>
__global__ void isosurface_extraction_kernel(
        const scalar_t* __restrict__ sdfs,
        const scalar_t* __restrict__ weights,
        const int64_t* __restrict__ grid_indices,
        const MiniVec<int, 3>* __restrict__ grid_coords_table,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_grid2grid,
        const MiniVec<int, 3>* __restrict__ cell_coords_table,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2cell,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2grid,
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
            neighbor_table_grid2grid[grid_idx];

    int cell_idx = dense_i;
    const MiniVec<int, 3>& cell_coord = cell_coords_table[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            neighbor_table_cell2cell[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            neighbor_table_cell2grid[cell_idx];

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
        const at::Tensor& neighbor_table_grid2grid,
        const at::Tensor& cell_coords_table,
        const at::Tensor& neighbor_table_cell2cell,  // (M^3, 8)
        const at::Tensor& neighbor_table_cell2grid,  // (M^3, 8)
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
                                neighbor_table_grid2grid.data_ptr()),
                        static_cast<MiniVec<int, 3>*>(
                                cell_coords_table.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_cell2cell.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_cell2grid.data_ptr()),
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
                                neighbor_table_grid2grid.data_ptr()),
                        static_cast<MiniVec<int, 3>*>(
                                cell_coords_table.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_cell2cell.data_ptr()),
                        static_cast<MiniVec<int64_t, 8>*>(
                                neighbor_table_cell2grid.data_ptr()),
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
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_grid2grid,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2cell,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2grid,
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
            neighbor_table_grid2grid[grid_idx];

    int cell_idx = dense_i;
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            neighbor_table_cell2cell[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            neighbor_table_cell2grid[cell_idx];

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
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_grid2grid,
        const MiniVec<int, 3>* __restrict__ cell_coords_table,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2cell,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2grid,
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
            neighbor_table_grid2grid[grid_idx];

    int cell_idx = dense_i;
    const MiniVec<int, 3>& cell_coord = cell_coords_table[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            neighbor_table_cell2cell[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            neighbor_table_cell2grid[cell_idx];

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
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_grid2grid,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2cell,
        const MiniVec<int64_t, 8>* __restrict__ neighbor_table_cell2grid,
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
            neighbor_table_grid2grid[grid_idx];

    int cell_idx = dense_i;
    const MiniVec<int64_t, 8>& neighbor_cell2cell =
            neighbor_table_cell2cell[cell_idx];
    const MiniVec<int64_t, 8>& neighbor_cell2grid =
            neighbor_table_cell2grid[cell_idx];

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

        const at::Tensor& grid_indices,              // (N, 1)
        const at::Tensor& grid_coords_table,         // (N, 3)
        const at::Tensor& neighbor_table_grid2grid,  // (N, 8) [-1 for
                                                     // invalid neighbors]
        const at::Tensor& cell_coords_table,         // (dense_res^3, 1)
        const at::Tensor& neighbor_table_cell2cell,  // (dense_res^3, 8)
        const at::Tensor& neighbor_table_cell2grid,  // (dense_res^3, 8)
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
                        neighbor_table_grid2grid.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2cell.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2grid.data_ptr()),
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
                        neighbor_table_grid2grid.data_ptr()),
                static_cast<MiniVec<int, 3>*>(cell_coords_table.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2cell.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2grid.data_ptr()),
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
                        neighbor_table_grid2grid.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2cell.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2grid.data_ptr()),
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
                        neighbor_table_grid2grid.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2cell.data_ptr()),
                static_cast<MiniVec<int64_t, 8>*>(
                        neighbor_table_cell2grid.data_ptr()),
                table_indices.data_ptr<int>(),
                static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
                triangle_counter.data_ptr<int>(),
                static_cast<MiniVec<int, 3>*>(triangles.data_ptr()), iso_value,
                weight_thr, grid_dim, num_cells_per_grid, len);
        C10_CUDA_CHECK(cudaDeviceSynchronize());
    });
    return {triangles, vertex_positions};
}
