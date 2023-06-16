#pragma once

#include <torch/extension.h>

// y = f(x, theta)
at::Tensor query_forward(
        const at::Tensor& embeddings,
        const at::Tensor& offsets,
        // queried x via a non-differentiable hash map lookup beforehand
        const at::Tensor& grid_indices,
        const at::Tensor& cell_indices,
        const at::Tensor& masks,
        // sparse luts
        const at::Tensor& neighbor_table_grid2grid,  // (N, 1)
        // dense luts
        const at::Tensor& neighbor_table_cell2cell,
        const at::Tensor& neighbor_table_cell2grid,
        const int64_t grid_dim,
        const std::string& interpolation);

// clang-format off
// Triggered both in forward (autograd.grad) and backward (loss.backward)
// z1, z2 = g(a, x, theta) = [a * df/dx, a * df/dtheta]
//
// when a=1,   w/o grad: z1, z2 = g(a, x) = [df/dx, df/dtheta]
// (to get df/dx with autograd.grad in a forward call)
// Note in this case, only z1 is used in the follow up computations (grad_x)
//
// when a=dl/dy, w/ grad: z1, z2 = g(a, x) = [dl/dy * df/dx, dl/dy * df/dtheta] = [dl/dx, dl/dtheta]
// (to backprop in the backward pass in a backward call)
// clang-format on
std::tuple<at::Tensor, at::Tensor> query_backward_forward(
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
        const int64_t grid_dim,
        const std::string& interpolation);

// clang-format off
// Triggered only in backward call w.r.t. z1=dfdx (loss.backward)
// w1, w2, w3 = h(b1=dl/dz1, b2=dl/dz2, x, theta)
// = [dlda = dl/dz1 dz1/da + dl/dz2 dz2/da,
//    dldx = dl/dz1 dz1/dx + dl/dz2 dz2/dx,
//    dldtheta = dl/dz1 dz1/dtheta + dl/dz2 dz2/dtheta]
//
// Remember only z1 is used in the forward computation (the gradient df/dx),
// we can safely ignore z2 b=dldz.
//
// The output is thus reduced to:
// w1 = dl/dz1 dz1/da = dl/dz1 d(a * df/dx)/da = dl/dz1 df/dx
// w2 = dl/dz1 dz1/dx = dl/dz1 d(a * df/dx)/dx = dl/dz1 d^2f/dx^2
// w3 = dl/dz1 dz1/dtheta = dl/dz1 d(a * df/dx)/dtheta = dl/dz1 d^2f/dxdtheta
//
// If we do not optimize x (that might be dependent on poses, for instance),
// we can safely ignore w2 as well.
// clang-format on
std::tuple<at::Tensor, at::Tensor, at::Tensor> query_backward_backward(
        const at::Tensor& grad_dLdembedding,
        const at::Tensor& grad_dLdoffset,
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
        const int64_t grid_dim,
        const std::string& interpolation);

at::Tensor isosurface_extraction(
        const at::Tensor& sdfs,
        const at::Tensor& weights,
        // query
        const at::Tensor& grid_indices,
        // sparse luts
        const at::Tensor& grid_coords_table,         // (N, 3)
        const at::Tensor& neighbor_table_grid2grid,  // (N, 1)
        // dense luts
        const at::Tensor& cell_coords_table,
        const at::Tensor& neighbor_table_cell2cell,  // (M^3, 8)
        const at::Tensor& neighbor_table_cell2grid,  // (M^3, 8)
        const int64_t grid_dim,
        const float iso_value,
        const float weight_thr);

std::tuple<at::Tensor, at::Tensor> marching_cubes(
        const at::Tensor& sdfs,
        const at::Tensor& weights,
        // query
        const at::Tensor& grid_indices,
        // sparse luts
        const at::Tensor& grid_coords_table,         // (N, 3)
        const at::Tensor& neighbor_table_grid2grid,  // (N, 1)
        // dense luts
        const at::Tensor& cell_coords_table,
        const at::Tensor& neighbor_table_cell2cell,  // (M^3, 8)
        const at::Tensor& neighbor_table_cell2grid,  // (M^3, 8)
        const int64_t grid_dim,
        const float iso_value,
        const float weight_thr);

// Only provide forward convolution (interpolation) for now
at::Tensor convolution_forward(
        const at::Tensor& embeddings,
        const at::Tensor& weights,                   // (K window)
        const at::Tensor& grid_indices,              // (N, 1)
        const at::Tensor& cell_indices,              // (M^3, 8)
        const at::Tensor& neighbor_table_grid2grid,  // (N, K^3)
        const at::Tensor& neighbor_table_cell2cell,  // (M^3, K^3)
        const at::Tensor& neighbor_table_cell2grid,  // (M^3, K^3)
        const int64_t grid_dim);
