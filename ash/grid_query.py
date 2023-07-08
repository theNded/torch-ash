from typing import Literal

import torch
import torch.nn as nn

from .common import _get_c_extension

backend = _get_c_extension()


class SparseDenseGridQuery(torch.autograd.Function):
    """Interpolate the embeddings.
    Each query point x can be located in a cell with grid_indices and cell_indices.
    The offset defined by its relative position to the cell corner gives interpolation ratio in the cell unit.

    Therefore the output is differentiable w.r.t. embeddings and offsets.
    ----------
    |     |  |
    |---->x  |
    |        |
    ----------
    """

    @staticmethod
    def forward(
        ctx,
        embeddings: torch.Tensor,
        offsets: torch.Tensor,
        grid_indices: torch.Tensor,
        cell_indices: torch.Tensor,
        masks: torch.Tensor,
        lut_grid_nb2grid_idx: torch.Tensor,
        lut_cell_nb2cell_idx: torch.Tensor,
        lut_cell_nb2grid_nb: torch.Tensor,
        grid_dim: int,
        interpolation: Literal["linear", "smooth_step"] = "smooth_step",
    ) -> torch.Tensor:
        """
        Forward pass of the interpolation.
        For simplicity, we only consider a single query point offset of (3,) and its 8 neighbors
        y = \\sum_{i=0}^7 weight(offset)[i] * embeddings[i]
        Args:
            embeddings: (num_embeddings, cells_per_grid, embedding_dim) embeddings of the grid [differentiable]
            offsets: (num_queries, 3) offsets of the input [differentiable]

            grid_indices: (num_queries, 1) grid index of the input
            cell_indices: (num_queries, 1) cell index of the input
            masks: (num_queries, 1) mask of the input

            lut_grid_nb2grid_idx: (num_embeddings, 8) precomputed neighbor table from grid index to grid index
            lut_cell_nb2cell_idx: (cells_per_grid, 8) precomputed neighbor table from cell index to cell index
            lut_cell_nb2grid_nb: (cells_per_grid, 8) precomputed neighbor table from cell index to grid index

            grid_dim: int cells_per_grid = grid_dim**3

        Returns:
            y: (num_queries, embedding_dim) interpolated embeddings
        """
        ctx.save_for_backward(
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            lut_grid_nb2grid_idx,
            lut_cell_nb2cell_idx,
            lut_cell_nb2grid_nb,
        )

        ctx.grid_dim = grid_dim
        ctx.interpolation = interpolation

        y = backend.query_forward(
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            lut_grid_nb2grid_idx,
            lut_cell_nb2cell_idx,
            lut_cell_nb2grid_nb,
            grid_dim,
            interpolation,
        )
        return y

    @staticmethod
    def backward(ctx, z: torch.Tensor):
        """Backward pass of the interpolation.
        Supports both forward (for explicit gradient computation via autograd.grad)
        and the conventional backward.
        Detailed in SparseDenseGridQueryBackward.
        """
        (
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            lut_grid_nb2grid_idx,
            lut_cell_nb2cell_idx,
            lut_cell_nb2grid_nb,
        ) = ctx.saved_tensors

        grad_embeddings, grad_offsets = SparseDenseGridQueryBackward.apply(
            z.contiguous(),
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            lut_grid_nb2grid_idx,
            lut_cell_nb2cell_idx,
            lut_cell_nb2grid_nb,
            ctx.grid_dim,
            ctx.interpolation,
        )

        return (
            grad_embeddings,
            grad_offsets,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SparseDenseGridQueryBackward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z,
        embeddings,
        offsets,
        grid_indices,
        cell_indices,
        masks,
        lut_grid_nb2grid_idx,
        lut_cell_nb2cell_idx,
        lut_cell_nb2grid_nb,
        grid_dim,
        interpolation,
    ):
        """
        Forward pass of the backward function.
        Args:
            z: (num_queries, embedding_dim) gradient of the output w.r.t. y

            z could be a tensor without gradient.
            Itself is the gradient of the loss w.r.t. y, i.e.,
                output = dL/dy * dy/dembeddings, dL/dy * dy/doffsets
                       = dL/dembeddings, dL/doffsets

            z could also be a tensor with gradient when used as the forward pass
            to compute jvp via torch.autograd.grad.
            It gradient should be passed back to downstream layers (e.g. MLP).

            Since y = \\sum_{i=0}^7 weight(offset)[i] * embeddings[i]

            grad_embeddings[i] = z * weight(offset)[i]
            grad_outputs = (z * embeddings[i]) * grad_weight(offset)[i]

        weight = [(1-x)(1-y)(1-z), x(1-y)(1-z), ..., xyz] (1 x 8)
        grad_weight = [[-(1-y)(1-z), (1-y)(1-z), ..., yz] (3 x 8)
                       [-(1-x)(1-z), -x(1-z), ...,    xz]
                       [-(1-x)(1-y), -x(1-y), ...,    xy]
        hessian_weight = [[0 1-z 1-y]            [0 z y]  ((3,3) x 8)
                          [1-z 0 1-x]            [z 0 x]
                          [1-y 1-x 0], ...,      [y x 0]]

        Returns:
            grad_embeddings: (num_embeddings, cells_per_grid, embedding_dim) gradient of the embeddings
            grad_offsets: (num_queries, 3)
        """
        ctx.save_for_backward(
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            lut_grid_nb2grid_idx,
            lut_cell_nb2cell_idx,
            lut_cell_nb2grid_nb,
        )
        ctx.grid_dim = grid_dim
        ctx.interpolation = interpolation

        grad_embeddings, grad_offsets = backend.query_backward_forward(
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            lut_grid_nb2grid_idx,
            lut_cell_nb2cell_idx,
            lut_cell_nb2grid_nb,
            grid_dim,
            interpolation,
        )

        return grad_embeddings, grad_offsets

    @staticmethod
    def backward(
        ctx, grad_grad_embeddings: torch.Tensor, grad_grad_offsets: torch.Tensor
    ):
        """Backward pass of the backward function.
        When a gradient is computed in by the backward's forward pass and used to compute a loss, its gradient
        need to be properly back propagated back to embeddings and offsets.

        Args:
            z: (num_queries, embedding_dim) gradient of the output w.r.t. y
            grad_grad_embeddings: (num_embeddings, cells_per_grid, embedding_dim) gradient of the embeddings
            grad_grad_offsets: (num_queries, 3)

            Let
                w1 = grad_embeddings[i] = z * weight(offset)[i], grad_w1 = grad_grad_embeddings[i]
                w2 = grad_offsets = (z * embeddings[i]) * grad_weight(offset)[i], grad_w2 = grad_grad_offsets

        TODO: now grad_grad_embeddings is not used. If feature-grid regularization is needed,
        we need to add it back. Now we safely ignore grad_w1.
        TODO: at current, grad_offsets are skipped as offsets are not optimized.
            Then we have
                grad_z = grad_w1 * dw1/dz + grad_w2 * dw2/dz = grad_w2 * dw2/dz
                       = (grad_w2 * grad_weight(offset)[i]) * embeddings[i] => (num_queries, embedding_dim)
                grad_embeddings[i] = grad_w2 * dw1/dembeddings[i]
                                   = (grad_w2  * grad_weight(offset)[i]) * z => (1, num_embeddings)
                grad_offsets = grad_w2 * dw2/doffsets
                             = (z * embeddings[i]) * grad_w2 * hessian_weight(offset)[i] => (1, 3)

        Returns:
            grad_z: (num_queries, embedding_dim) gradient of the output w.r.t. z
            grad_embeddings: (num_embeddings, cells_per_grid, embedding_dim) gradient of the embeddings
            grad_offsets: (num_queries, 3)
        """

        (
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            lut_grid_nb2grid_idx,
            lut_cell_nb2cell_idx,
            lut_cell_nb2grid_nb,
        ) = ctx.saved_tensors

        grad_z, grad_embeddings, grad_offsets = backend.query_backward_backward(
            grad_grad_embeddings,
            grad_grad_offsets,
            z,
            embeddings,
            offsets,
            grid_indices,
            cell_indices,
            masks,
            lut_grid_nb2grid_idx,
            lut_cell_nb2cell_idx,
            lut_cell_nb2grid_nb,
            ctx.grid_dim,
            ctx.interpolation,
        )

        return (
            grad_z,
            grad_embeddings,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
