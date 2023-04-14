#include <c10/cuda/CUDAException.h>

#include "mc_macros.h"
#include "minivec.h"
#include "sparsedense_grid.h"

template <typename scalar_t>
__global__ void query_forward_kernel(
        const scalar_t* embeddings,
        const MiniVec<float, 3>* offsets,
        const int64_t* sparse_indices,
        const int64_t* dense_indices,
        const MiniVec<int, 3>* dense_coords,
        const bool* masks,
        const MiniVec<int64_t, 8>* sparse_neighbor_indices_table,
        const MiniVec<int64_t, 8>* dense_neighbor_indices_table,
        scalar_t* output,
        const int64_t dense_grid_dim,
        const int64_t cells_per_grid,
        const int64_t embedding_dims,
        const int64_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len || !masks[i]) {
        return;
    }

    int sparse_index = sparse_indices[i];
    int dense_index = dense_indices[i];
    MiniVec<int, 3> dense_coord = dense_coords[i];
    MiniVec<float, 3> offset = offsets[i];

    MiniVec<int64_t, 8> sparse_neighbors =
            sparse_neighbor_indices_table[sparse_index];
    MiniVec<int64_t, 8> dense_neighbors =
            dense_neighbor_indices_table[dense_index];

    // TODO: dispatch or better serialization
    MiniVec<scalar_t, 16> sum_output = MiniVec<scalar_t, 16>::zeros();
    float sum_weight = 0.0;
    for (int nb = 0; nb < 8; ++nb) {
        int sparse_nb = 0;

        float weight = 1.0;
        for (int d = 0; d < 3; ++d) {
            int dim_code = (nb >> d) & 1;
            sparse_nb = (dense_coord[d] + (dim_code) == dense_grid_dim)
                                ? (sparse_nb | (1 << d))
                                : sparse_nb;
            weight *= (dim_code) ? (offset[d]) : (1 - offset[d]);
        }

        int sparse_nb_index = sparse_neighbors[sparse_nb];
        if (sparse_nb_index == -1) {
            continue;
        }
        int dense_nb_index = dense_neighbors[nb];

        int base_index = (sparse_nb_index * cells_per_grid + dense_nb_index) *
                         embedding_dims;
        for (int k = 0; k < embedding_dims; ++k) {
            sum_output[k] += weight * embeddings[base_index + k];
        }
        sum_weight += weight;
    }

    for (int k = 0; k < embedding_dims; ++k) {
        output[i * embedding_dims + k] = sum_output[k] / sum_weight;
    }
}

at::Tensor query_forward(
        const at::Tensor&
                embeddings,  // (num_embeddings, dense_res^3, embedding_dims)
        const at::Tensor& offsets,
        const at::Tensor& sparse_indices,
        const at::Tensor& dense_indices,
        const at::Tensor& dense_coords,
        const at::Tensor& masks,
        const at::Tensor& sparse_neighbor_indices_table,
        const at::Tensor& dense_neighbor_indices_table,
        const int64_t dense_grid_dim) {
    // TODO: wise block-thread unrolling
    const int64_t len = sparse_indices.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    const int64_t embedding_dims = embeddings.size(2);
    const int64_t cells_per_dense_grid = embeddings.size(1);

    at::Tensor output = at::zeros({len, embedding_dims}, embeddings.options());

    query_forward_kernel<float><<<blocks, threads>>>(
            embeddings.data_ptr<float>(),
            static_cast<MiniVec<float, 3>*>(offsets.data_ptr()),
            sparse_indices.data_ptr<int64_t>(),
            dense_indices.data_ptr<int64_t>(),
            static_cast<MiniVec<int, 3>*>(dense_coords.data_ptr()),
            masks.data_ptr<bool>(),
            static_cast<MiniVec<int64_t, 8>*>(
                    sparse_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    dense_neighbor_indices_table.data_ptr()),
            output.data_ptr<float>(), dense_grid_dim, cells_per_dense_grid,
            embedding_dims, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return output;
}

template <typename scalar_t>
__global__ void query_backward_forward_kernel(
        const scalar_t* z,
        const scalar_t* embeddings,
        const MiniVec<float, 3>* offsets,
        const int64_t* sparse_indices,
        const int64_t* dense_indices,
        const MiniVec<int, 3>* dense_coords,
        const bool* masks,
        const MiniVec<int64_t, 8>* sparse_neighbor_indices_table,
        const MiniVec<int64_t, 8>* dense_neighbor_indices_table,
        scalar_t* dLdembedding,
        MiniVec<float, 3>* dLdoffsets,
        const int64_t dense_grid_dim,
        const int64_t cells_per_grid,
        const int64_t embedding_dims,
        const int64_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len || !masks[i]) {
        return;
    }

    int sparse_index = sparse_indices[i];
    int dense_index = dense_indices[i];
    MiniVec<int, 3> dense_coord = dense_coords[i];
    MiniVec<float, 3> offset = offsets[i];

    MiniVec<int64_t, 8> sparse_neighbors =
            sparse_neighbor_indices_table[sparse_index];
    MiniVec<int64_t, 8> dense_neighbors =
            dense_neighbor_indices_table[dense_index];

    float sum_weight = 0.0;
    for (int nb = 0; nb < 8; ++nb) {
        int sparse_nb = 0;

        float weight = 1.0;
        MiniVec<float, 3> weight_grad = MiniVec<float, 3>::zeros();
        for (int d = 0; d < 3; ++d) {
            int dim_code = (nb >> d) & 1;
            sparse_nb = (dense_coord[d] + (dim_code) == dense_grid_dim)
                                ? (sparse_nb | (1 << d))
                                : sparse_nb;
            float w = (dim_code) ? (offset[d]) : (1 - offset[d]);
            float dw = (dim_code) ? (1) : (-1);
            weight *= w;
        }
        int sparse_nb_index = sparse_neighbors[sparse_nb];
        if (sparse_nb_index == -1) {
            continue;
        }
        sum_weight += weight;
    }

    for (int nb = 0; nb < 8; ++nb) {
        int sparse_nb = 0;

        float weight = 1.0;
        MiniVec<float, 3> weight_grad = MiniVec<float, 3>::zeros();
        for (int d = 0; d < 3; ++d) {
            int dim_code = (nb >> d) & 1;
            sparse_nb = (dense_coord[d] + (dim_code) == dense_grid_dim)
                                ? (sparse_nb | (1 << d))
                                : sparse_nb;
            float w = (dim_code) ? (offset[d]) : (1 - offset[d]);
            float dw = (dim_code) ? (1) : (-1);
            weight *= w;

            weight_grad[0] *= (d == 0) ? 1 : dw;
            weight_grad[1] *= (d == 1) ? 1 : dw;
            weight_grad[2] *= (d == 2) ? 1 : dw;
        }

        int sparse_nb_index = sparse_neighbors[sparse_nb];
        if (sparse_nb_index == -1) {
            continue;
        }
        int dense_nb_index = dense_neighbors[sparse_nb];

        int base_index = (sparse_nb_index * cells_per_grid + dense_nb_index) *
                         embedding_dims;

        float dot = 0.0;
        float normalized_weight = weight / sum_weight;
        for (int k = 0; k < embedding_dims; ++k) {
            atomicAdd(&dLdembedding[base_index + k],
                      normalized_weight * z[i * embedding_dims + k]);
            dot += embeddings[base_index + k] * z[i * embedding_dims + k];
        }
        dLdoffsets[i] += (dot * weight_grad);
    }
};

std::tuple<at::Tensor, at::Tensor> query_backward_forward(
        const at::Tensor& z,
        const at::Tensor& embeddings,
        const at::Tensor& offsets,
        const at::Tensor& sparse_indices,
        const at::Tensor& dense_indices,
        const at::Tensor& dense_coords,
        const at::Tensor& masks,
        const at::Tensor& sparse_neighbor_indices_table,
        const at::Tensor& dense_neighbor_indices_table,
        const int64_t dense_grid_dim) {
    const int64_t len = sparse_indices.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    const int64_t embedding_dims = embeddings.size(2);
    const int64_t cells_per_dense_grid = embeddings.size(1);

    at::Tensor dLdembedding = at::zeros_like(embeddings);
    at::Tensor dLdoffsets = at::zeros_like(offsets);

    // std::cout << "z in backward forward:" << z << std::endl;
    // std::cout << "embedding in backward forward:" << embeddings << std::endl;
    query_backward_forward_kernel<float><<<blocks, threads>>>(
            z.data_ptr<float>(), embeddings.data_ptr<float>(),
            static_cast<MiniVec<float, 3>*>(offsets.data_ptr()),
            sparse_indices.data_ptr<int64_t>(),
            dense_indices.data_ptr<int64_t>(),
            static_cast<MiniVec<int, 3>*>(dense_coords.data_ptr()),
            masks.data_ptr<bool>(),
            static_cast<MiniVec<int64_t, 8>*>(
                    sparse_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    dense_neighbor_indices_table.data_ptr()),
            dLdembedding.data_ptr<float>(),
            static_cast<MiniVec<float, 3>*>(dLdoffsets.data_ptr()),
            dense_grid_dim, cells_per_dense_grid, embedding_dims, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return std::make_tuple(dLdembedding, dLdoffsets);
}

template <typename scalar_t>
__global__ void isosurface_extraction_kernel(
        const scalar_t* sdfs,
        const scalar_t* weights,
        const int64_t* sparse_indices,
        const MiniVec<int, 3>* sparse_coords_table,
        const MiniVec<int64_t, 8>* sparse_neighbor_indices_table,
        const MiniVec<int, 3>* dense_coords_table,
        const MiniVec<int64_t, 8>* dense_neighbor_indices_table,
        int* output_counter,
        MiniVec<float, 3>* output_positions,
        const float iso_value,
        const float weight_thr,
        const int dense_grid_dim,
        const int cells_per_dense_grid,
        const int len) {
    const int kDim = 3;
    const int kNbs = 8;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) {
        return;
    }

    int sparse_i = i / cells_per_dense_grid;
    int dense_i = i % cells_per_dense_grid;

    int sparse_index = sparse_indices[sparse_i];
    MiniVec<int, 3> sparse_coord = sparse_coords_table[sparse_index];
    MiniVec<int64_t, 8> sparse_neighbors =
            sparse_neighbor_indices_table[sparse_index];
    if (sparse_index != sparse_neighbors[0] or sparse_index == -1) {
        printf("extract isosurfaces: should never reach here!\n");
        return;
    }

    int dense_index = dense_i;
    MiniVec<int, 3> dense_coord = dense_coords_table[dense_index];
    MiniVec<int64_t, 8> dense_neighbors =
            dense_neighbor_indices_table[dense_index];

    scalar_t self_sdf = sdfs[sparse_index * cells_per_dense_grid + dense_index];
    scalar_t self_weight =
            weights[sparse_index * cells_per_dense_grid + dense_index];
    scalar_t self_isodiff = self_sdf - iso_value;

    if (self_weight < weight_thr) {
        return;
    }

    MiniVec<float, 3> offset = MiniVec<float, 3>::zeros();
    for (int d = 0; d < kDim; ++d) {
        // Look for 3 neighbors in each dimension
        offset[d] = 1;

        int sparse_nb = (dense_coord[d] + 1 == dense_grid_dim) ? (1 << d) : 0;
        int dense_nb = 1 << d;

        int sparse_nb_index = sparse_neighbors[sparse_nb];
        int dense_nb_index = dense_neighbors[dense_nb];

        if (sparse_nb_index == -1) {
            return;
        }
        scalar_t nb_sdf =
                sdfs[sparse_nb_index * cells_per_dense_grid + dense_nb_index];
        scalar_t nb_weight = weights[sparse_nb_index * cells_per_dense_grid +
                                     dense_nb_index];
        scalar_t nb_isodiff = nb_sdf - iso_value;

        if (self_isodiff * nb_isodiff < 0 && nb_weight >= weight_thr) {
            int output_index = atomicAdd(output_counter, 1);

            if (output_positions) {
                float ratio = self_isodiff / (self_isodiff - nb_isodiff);
                MiniVec<int, 3> self_position =
                        sparse_coord * dense_grid_dim + dense_coord;
                MiniVec<float, 3> surface_position =
                        self_position.template cast<float>() + ratio * offset;
                output_positions[output_index] = surface_position;
            }
        }

        offset[d] = 0;
    }
}

at::Tensor isosurface_extraction(
        const at::Tensor& sdfs,     // (num_embeddings, dense_res^3, 1)
        const at::Tensor& weights,  // (num_embeddings, dense_res^3, 1)
        const at::Tensor& sparse_indices,
        const at::Tensor& sparse_coords_table,
        const at::Tensor& sparse_neighbor_indices_table,
        const at::Tensor& dense_coords_table,
        const at::Tensor& dense_neighbor_indices_table,  // (M^3, 8)
        const int64_t dense_grid_dim,
        const float iso_value,
        const float weight_thr) {
    // TODO: wise block-thread unrolling
    const int64_t len = sparse_indices.size(0) * dense_coords_table.size(0);

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    at::Tensor output_counter = at::zeros({}, sparse_coords_table.options());

    int cells_per_dense_grid = sdfs.size(1);

    isosurface_extraction_kernel<float><<<blocks, threads>>>(
            sdfs.data_ptr<float>(), weights.data_ptr<float>(),
            sparse_indices.data_ptr<int64_t>(),
            static_cast<MiniVec<int, 3>*>(sparse_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    sparse_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int, 3>*>(dense_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    dense_neighbor_indices_table.data_ptr()),
            output_counter.data_ptr<int>(), nullptr, iso_value, weight_thr,
            dense_grid_dim, cells_per_dense_grid, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    at::Tensor output_positions =
            at::zeros({output_counter.item<int>(), 3}, sdfs.options());
    output_counter = at::zeros({}, sparse_coords_table.options());
    isosurface_extraction_kernel<float><<<blocks, threads>>>(
            sdfs.data_ptr<float>(), weights.data_ptr<float>(),
            sparse_indices.data_ptr<int64_t>(),
            static_cast<MiniVec<int, 3>*>(sparse_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    sparse_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int, 3>*>(dense_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    dense_neighbor_indices_table.data_ptr()),
            output_counter.data_ptr<int>(),
            static_cast<MiniVec<float, 3>*>(output_positions.data_ptr()),
            iso_value, weight_thr, dense_grid_dim, cells_per_dense_grid, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return output_positions;
}

template <typename scalar_t>
__global__ void marching_cubes_table_index_kernel(
        const scalar_t* sdfs,
        const scalar_t* weights,
        const int64_t* sparse_indices,
        const MiniVec<int, 3>* sparse_coords_table,
        const MiniVec<int64_t, 8>* sparse_neighbor_indices_table,
        const MiniVec<int, 3>* dense_coords_table,
        const MiniVec<int64_t, 8>* dense_neighbor_indices_table,
        int* table_indices,
        MiniVec<int, 3>*
                edge_vertex_indices,  // 3 per cell in the sparse-dense grid
        const float iso_value,
        const float weight_thr,
        const int dense_grid_dim,
        const int cells_per_dense_grid,
        const int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    int sparse_i = i / cells_per_dense_grid;
    int dense_i = i % cells_per_dense_grid;

    int sparse_index = sparse_indices[sparse_i];
    MiniVec<int, 3> sparse_coord = sparse_coords_table[sparse_index];
    MiniVec<int64_t, 8> sparse_neighbors =
            sparse_neighbor_indices_table[sparse_index];

    int dense_index = dense_i;
    MiniVec<int, 3> dense_coord = dense_coords_table[dense_index];
    MiniVec<int64_t, 8> dense_neighbors =
            dense_neighbor_indices_table[dense_index];

    scalar_t self_sdf = sdfs[sparse_index * cells_per_dense_grid + dense_index];
    scalar_t self_weight =
            weights[sparse_index * cells_per_dense_grid + dense_index];
    scalar_t self_isodiff = self_sdf - iso_value;

    if (self_weight < weight_thr) {
        return;
    }

    int table_index = 0;
    for (int corner = 0; corner < 8; ++corner) {
        int sparse_nb = 0;

        int dense_nb = nb_mc_vtx_map[corner];
        for (int d = 0; d < 3; ++d) {
            sparse_nb =
                    (dense_coord[d] + ((dense_nb >> d) & 1) == dense_grid_dim)
                            ? (sparse_nb | (1 << d))
                            : sparse_nb;
        }

        int sparse_nb_index = sparse_neighbors[sparse_nb];
        int dense_nb_index = dense_neighbors[dense_nb];

        if (sparse_nb_index == -1) {
            return;
        }

        scalar_t nb_sdf =
                sdfs[sparse_nb_index * cells_per_dense_grid + dense_nb_index];
        scalar_t nb_weight = weights[sparse_nb_index * cells_per_dense_grid +
                                     dense_nb_index];
        scalar_t nb_isodiff = nb_sdf - iso_value;

        if (nb_weight < weight_thr) {
            return;
        }

        table_index |= ((nb_isodiff < 0) << corner);
    }

    // No vertex, return
    if (table_index == 0 || table_index == 255) {
        return;
    }
    table_indices[sparse_index * cells_per_dense_grid + dense_index] =
            table_index;

    // 1st 12 bits encodes edge-isosurface intersection
    int edges_encoding = edge_table[table_index];
    for (int edge = 0; edge < 12; ++edge) {
        if (!(edges_encoding & (1 << edge))) {
            continue;
        }

        // A cell is in charge of 3 edges at xyz directions
        // First decide the cell in charge of this edge (12 edges for a cell,
        // could be managed by a neighbor)
        // and the edge direction
        int nb = nb_mc_vtx_map[edge_to_vert[edge][0]];  // [0, 8)
        int edge_dir = edge_shifts[edge][3];            // [0, 1, 2]

        int sparse_nb = 0;
        for (int d = 0; d < 3; ++d) {
            sparse_nb = (dense_coord[d] + ((nb >> d) & 1) == dense_grid_dim)
                                ? (sparse_nb | (1 << d))
                                : sparse_nb;
        }
        int dense_nb = nb;

        // must be valid from the table decision
        int sparse_nb_index = sparse_neighbors[sparse_nb];
        int dense_nb_index = dense_neighbors[dense_nb];
        if (sparse_nb_index == -1) {
            printf("Should never reach here!\n");
            return;
        }

        // Placeholder for the vertex index
        edge_vertex_indices[sparse_nb_index * cells_per_dense_grid +
                            dense_nb_index][edge_dir] = -1;
    }
}

template <typename scalar_t>
__global__ void marching_cubes_vertex_kernel(
        const scalar_t* sdfs,
        const scalar_t* weights,
        const int64_t* sparse_indices,
        const MiniVec<int, 3>* sparse_coords_table,
        const MiniVec<int64_t, 8>* sparse_neighbor_indices_table,
        const MiniVec<int, 3>* dense_coords_table,
        const MiniVec<int64_t, 8>* dense_neighbor_indices_table,
        MiniVec<int, 3>*
                edge_vertex_indices,  // 3 per cell in the sparse-dense grid
        int* vertex_counter,
        MiniVec<float, 3>* vertex_positions,
        const float iso_value,
        const float weight_thr,
        const int dense_grid_dim,
        const int cells_per_dense_grid,
        const int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    int sparse_i = i / cells_per_dense_grid;
    int dense_i = i % cells_per_dense_grid;

    int sparse_index = sparse_indices[sparse_i];
    MiniVec<int, 3> sparse_coord = sparse_coords_table[sparse_index];
    MiniVec<int64_t, 8> sparse_neighbors =
            sparse_neighbor_indices_table[sparse_index];

    int dense_index = dense_i;
    MiniVec<int, 3> dense_coord = dense_coords_table[dense_index];
    MiniVec<int64_t, 8> dense_neighbors =
            dense_neighbor_indices_table[dense_index];

    scalar_t self_sdf = sdfs[sparse_index * cells_per_dense_grid + dense_index];
    scalar_t self_weight =
            weights[sparse_index * cells_per_dense_grid + dense_index];
    scalar_t self_isodiff = self_sdf - iso_value;

    if (self_weight < weight_thr) {
        return;
    }

    MiniVec<float, 3> offset = MiniVec<float, 3>::zeros();
    for (int d = 0; d < 3; ++d) {
        if (edge_vertex_indices[sparse_index * cells_per_dense_grid +
                                dense_index][d] != -1) {
            continue;
        }
        offset[d] = 1;

        int sparse_nb = (dense_coord[d] + 1 == dense_grid_dim) ? (1 << d) : 0;
        int dense_nb = 1 << d;

        int sparse_nb_index = sparse_neighbors[sparse_nb];
        int dense_nb_index = dense_neighbors[dense_nb];

        if (sparse_nb_index == -1) {
            printf("should never reach here!\n");
            return;
        }
        scalar_t nb_sdf =
                sdfs[sparse_nb_index * cells_per_dense_grid + dense_nb_index];
        scalar_t nb_isodiff = nb_sdf - iso_value;

        int output_index = atomicAdd(vertex_counter, 1);

        float ratio = self_isodiff / (self_isodiff - nb_isodiff);
        MiniVec<int, 3> self_position =
                sparse_coord * dense_grid_dim + dense_coord;
        MiniVec<float, 3> surface_position =
                self_position.template cast<float>() + ratio * offset;
        edge_vertex_indices[sparse_index * cells_per_dense_grid + dense_index]
                           [d] = output_index;
        offset[d] = 0;
        vertex_positions[output_index] = surface_position;
    }
}

template <typename scalar_t>
__global__ void marching_cubes_triangle_kernel(
        scalar_t* sdfs,
        const scalar_t* weights,
        const int64_t* sparse_indices,
        const MiniVec<int, 3>* sparse_coords_table,
        const MiniVec<int64_t, 8>* sparse_neighbor_indices_table,
        const MiniVec<int, 3>* dense_coords_table,
        const MiniVec<int64_t, 8>* dense_neighbor_indices_table,
        const int* table_indices,
        const MiniVec<int, 3>*
                edge_vertex_indices,  // 3 per cell in the sparse-dense grid
        int* triangle_counter,
        MiniVec<int, 3>* triangle_indices,
        const float iso_value,
        const float weight_thr,
        const int dense_grid_dim,
        const int cells_per_dense_grid,
        const int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= len) return;

    int sparse_i = i / cells_per_dense_grid;
    int dense_i = i % cells_per_dense_grid;

    int sparse_index = sparse_indices[sparse_i];
    MiniVec<int, 3> sparse_coord = sparse_coords_table[sparse_index];
    MiniVec<int64_t, 8> sparse_neighbors =
            sparse_neighbor_indices_table[sparse_index];

    int dense_index = dense_i;
    MiniVec<int, 3> dense_coord = dense_coords_table[dense_index];
    MiniVec<int64_t, 8> dense_neighbors =
            dense_neighbor_indices_table[dense_index];

    int table_index =
            table_indices[sparse_index * cells_per_dense_grid + dense_index];
    if (table_index == 0) {
        return;
    }
    for (int tri = 0; tri < 16; tri += 3) {
        if (tri_table[table_index][tri] == -1) {
            return;
        }
        int tri_idx = atomicAdd(triangle_counter, 1);

        if (triangle_indices == nullptr) continue;

        for (int vertex = 0; vertex < 3; ++vertex) {
            int edge = tri_table[table_index][tri + vertex];

            // extract the vertex index from the edge
            int sparse_nb = 0;
            int dense_nb = nb_mc_vtx_map[edge_to_vert[edge][0]];
            int edge_dir = edge_shifts[edge][3];
            for (int d = 0; d < 3; ++d) {
                sparse_nb = (dense_coord[d] + ((dense_nb >> d) & 1) ==
                             dense_grid_dim)
                                    ? (sparse_nb | (1 << d))
                                    : sparse_nb;
            }
            int sparse_nb_index = sparse_neighbors[sparse_nb];
            int dense_nb_index = dense_neighbors[dense_nb];

            if (sparse_nb_index == -1) {
                printf("triangle extraction: should never reach here\n");
                return;
            }

            int vertex_idx =
                    edge_vertex_indices[sparse_nb_index * cells_per_dense_grid +
                                        dense_nb_index][edge_dir];

            triangle_indices[tri_idx][(2 - vertex)] = vertex_idx;
        }
    }
}

std::tuple<at::Tensor, at::Tensor> marching_cubes(
        const at::Tensor& sdfs,     // (num_embeddings, dense_res^3, 1)
        const at::Tensor& weights,  // (num_embeddings, dense_res^3, 1)

        const at::Tensor& sparse_indices,                 // (N, 1)
        const at::Tensor& sparse_coords_table,            // (N, 3)
        const at::Tensor& sparse_neighbor_indices_table,  // (N, 8) [-1 for
                                                          // invalid neighbors]
        const at::Tensor& dense_coords_table,             // (dense_res^3, 1)
        const at::Tensor& dense_neighbor_indices_table,   // (dense_res^3, 8)
        const int64_t dense_grid_dim,
        const float iso_value,
        const float weight_thr) {
    const int64_t num_embeddings = sdfs.size(0);
    const int64_t cells_per_dense_grid = sdfs.size(1);

    const int64_t num_sparse_grids = sparse_indices.size(0);
    const int64_t len = num_sparse_grids * cells_per_dense_grid;

    const int64_t threads = 256;
    const int64_t blocks = (len + threads - 1) / threads;

    std::cout << "num_embeddings: " << num_embeddings << std::endl;
    std::cout << "cells_per_dense_grid: " << cells_per_dense_grid << std::endl;
    std::cout << "num_sparse_grids: " << num_sparse_grids << std::endl;

    // Determine marching cubes table index and vertex existence per cell
    auto options =
            at::TensorOptions().dtype(torch::kInt32).device(sdfs.device());

    at::Tensor table_indices =
            at::zeros({num_embeddings, cells_per_dense_grid}, options);
    at::Tensor edge_vertex_indices =
            at::zeros({num_embeddings, cells_per_dense_grid, 3}, options);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    printf("Before calling marching_cubes_table_index_kernel\n");

    marching_cubes_table_index_kernel<float><<<blocks, threads>>>(
            sdfs.data_ptr<float>(), weights.data_ptr<float>(),
            sparse_indices.data_ptr<int64_t>(),
            static_cast<MiniVec<int, 3>*>(sparse_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    sparse_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int, 3>*>(dense_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    dense_neighbor_indices_table.data_ptr()),
            table_indices.data_ptr<int>(),
            static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
            iso_value, weight_thr, dense_grid_dim, cells_per_dense_grid, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    printf("After calling marching_cubes_table_indices_kernel\n");
    int num_vertices = edge_vertex_indices.eq(-1).sum().item<int>();
    std::cout << num_vertices << std::endl;
    at::Tensor vertex_counter = at::zeros({}, options);
    at::Tensor vertex_positions = at::zeros({num_vertices, 3}, sdfs.options());
    // return std::tuple<at::Tensor, at::Tensor>(at::zeros({}),
    // vertex_positions);

    // Determine vertex positions
    printf("Before calling marching_cubes_vertex_kernel\n");
    marching_cubes_vertex_kernel<float><<<blocks, threads>>>(
            sdfs.data_ptr<float>(), weights.data_ptr<float>(),
            sparse_indices.data_ptr<int64_t>(),
            static_cast<MiniVec<int, 3>*>(sparse_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    sparse_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int, 3>*>(dense_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    dense_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
            vertex_counter.data_ptr<int>(),
            static_cast<MiniVec<float, 3>*>(vertex_positions.data_ptr()),
            iso_value, weight_thr, dense_grid_dim, cells_per_dense_grid, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "vertex_counter: " << vertex_counter.item<int>() << std::endl;

    at::Tensor triangle_counter = at::zeros({}, options);
    printf("pass 1: triangle kernel\n");
    marching_cubes_triangle_kernel<float><<<blocks, threads>>>(
            sdfs.data_ptr<float>(), weights.data_ptr<float>(),
            sparse_indices.data_ptr<int64_t>(),
            static_cast<MiniVec<int, 3>*>(sparse_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    sparse_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int, 3>*>(dense_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    dense_neighbor_indices_table.data_ptr()),
            table_indices.data_ptr<int>(),
            static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
            triangle_counter.data_ptr<int>(), nullptr, iso_value, weight_thr,
            dense_grid_dim, cells_per_dense_grid, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    printf("pass 2: triangle kernel\n");
    at::Tensor triangles =
            at::zeros({triangle_counter.item<int>(), 3}, options);
    triangle_counter.zero_();
    marching_cubes_triangle_kernel<float><<<blocks, threads>>>(
            sdfs.data_ptr<float>(), weights.data_ptr<float>(),
            sparse_indices.data_ptr<int64_t>(),
            static_cast<MiniVec<int, 3>*>(sparse_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    sparse_neighbor_indices_table.data_ptr()),
            static_cast<MiniVec<int, 3>*>(dense_coords_table.data_ptr()),
            static_cast<MiniVec<int64_t, 8>*>(
                    dense_neighbor_indices_table.data_ptr()),
            table_indices.data_ptr<int>(),
            static_cast<MiniVec<int, 3>*>(edge_vertex_indices.data_ptr()),
            triangle_counter.data_ptr<int>(),
            static_cast<MiniVec<int, 3>*>(triangles.data_ptr()), iso_value,
            weight_thr, dense_grid_dim, cells_per_dense_grid, len);
    C10_CUDA_CHECK(cudaDeviceSynchronize());

    return {triangles, vertex_positions};
}
