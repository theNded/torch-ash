import torch

def enumerate_neighbor_coord_offsets(
    dim: int, radius: int, bidirectional: bool
) -> torch.Tensor:
    """Generate neighbor coordinate offsets.
    This function is independent of the choice of grid or cell.
    In the 1-radius, non-bidirectional case, it is equivalent to morton code
    i.e., 001, 010, 011, 100, 101, 110, 111 (3 digits in zyx order)
    Args:
        dim: dimension of the coordinate
        radius: radius of the neighborhood
        bidirectional: whether to include negative offsets.
    Returns:
        If bidirectional: enumerating [-r, -r, -r] -- [r, r, r]
          returns a tensor of shape ((2 * radius + 1) ** dim, dim)
        If not bidirectional: enumerating [0, 0, 0] -- [r, r, r]
          returns a tensor of shape ((radius + 1) ** dim, dim)
    """
    if bidirectional:
        arange = torch.arange(-radius, radius + 1)
    else:
        arange = torch.arange(0, radius + 1)

    idx2offset = (
        torch.stack(torch.meshgrid(*[arange for _ in range(dim)], indexing="ij"))
        .reshape(dim, -1)
        .T.flip(dims=(1,))  # zyx order => xyz for easier adding
        .contiguous()
    )

    def fn_offset2idx(offset):
        """
        offset: (N, 3)
        returns idx: (N, 1)
        """
        assert len(offset.shape) == 2

        padding = radius if bidirectional else 0
        window = 2 * radius + 1 if bidirectional else radius + 1
        idx = torch.zeros_like(offset[:, 0])
        for i in range(dim - 1, -1, -1):
            idx = (offset[:, i] + padding) + idx * window
        return idx

    return idx2offset, fn_offset2idx

idx2offset, fn_offset2idx = enumerate_neighbor_coord_offsets(3, 1, False)
indices = torch.randint(len(idx2offset), (8,))
print(indices)
print(idx2offset[indices])
print(fn_offset2idx(idx2offset[indices]))
