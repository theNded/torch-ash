#pragma once

#include <torch/extension.h>

// same pattern could be applied to query (?)
// input: sparse_indices, dense_indices, interp ratios
// output: interpolated embeddings

// f(x, embeddings) => f(interp_ratios, embeddings) => features
std::tuple<at::Tensor, at::Tensor> forward(
        const at::Tensor& embeddings,
        // queried x via a non-differentiable hash map lookup beforehand
        const at::Tensor& sparse_indices,
        const at::Tensor& dense_indices,
        const at::Tensor& interp_ratios,
        const at::Tensor& masks,
        // sparse luts
        const at::Tensor& sparse_coords_table,            // (N, 3)
        const at::Tensor& sparse_neighbor_indices_table,  // (N, 1)
        // dense luts
        const at::Tensor& dense_coords_table,
        const at::Tensor& dense_neighbor_indices_table,  // (M^3, 8)
        const int64_t dense_grid_dim);

// df(dLdf, d^2Ldf^2)
// => d embeddings passed to embeddings, d interp_ratios passed to dx
at::Tensor backward(
        const at::Tensor& grad_output_1st,
        const at::Tensor& grad_output_2nd,
        // queried x via a non-differentiable hash map lookup beforehand
        const at::Tensor& sparse_indices,
        const at::Tensor& dense_indices,
        const at::Tensor& interp_ratios,
        const at::Tensor& masks,
        // sparse luts
        const at::Tensor& sparse_coords_table,            // (N, 3)
        const at::Tensor& sparse_neighbor_indices_table,  // (N, 1)
        // dense luts
        const at::Tensor& dense_coords_table,
        const at::Tensor& dense_neighbor_indices_table,  // (M^3, 8)
        const int64_t dense_grid_dim);

at::Tensor isosurface_extraction(
        const at::Tensor& sdfs,
        const at::Tensor& weights,
        // query
        const at::Tensor& sparse_indices,
        // sparse luts
        const at::Tensor& sparse_coords_table,            // (N, 3)
        const at::Tensor& sparse_neighbor_indices_table,  // (N, 1)
        // dense luts
        const at::Tensor& dense_coords_table,
        const at::Tensor& dense_neighbor_indices_table,  // (M^3, 8)
        const int64_t dense_grid_dim,
        const float iso_value,
        const float weight_thr);

std::tuple<at::Tensor, at::Tensor> marching_cubes(
        const at::Tensor& sdfs,
        const at::Tensor& weights,
        // query
        const at::Tensor& sparse_indices,
        // sparse luts
        const at::Tensor& sparse_coords_table,            // (N, 3)
        const at::Tensor& sparse_neighbor_indices_table,  // (N, 1)
        // dense luts
        const at::Tensor& dense_coords_table,
        const at::Tensor& dense_neighbor_indices_table,  // (M^3, 8)
        const int64_t dense_grid_dim,
        const float iso_value,
        const float weight_thr);
