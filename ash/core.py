from typing import Union, Tuple, Dict, OrderedDict, Optional

from collections import OrderedDict
import warnings

import torch
import torch.nn as nn

from .common import _get_c_extension

backend = _get_c_extension()


"""
The core differences between collision-free (ASH) and collision-allowed (NGP) hash map are:
1. Collision-free requires spatial initialization, while collision-allowed does not.
2. A query could fail in collision-free, while it will always succeed in collision-allowed.

In other words, forward function for collision-free modules will always return a result with a mask.
This is acceptable (and even desirable) for classical operations:
- Marching Cubes
- TSDF fusion
- Sampling

This is problematic for (differentiable) neural operations:
- Encoding
One work around would be using an padding_index=capacity and return encoding at this exact index
    feat_fg, mask_fg = ash_module.forward(x)

Another work around for this could be an additional low-resolution dense
"empty space grid" equipped with contraction, which guarantees a valid result.
The interface would be similar to:
    feat_fg, mask_fg = ash_module.forward(x)
    feat_bg = empty_space_module.forward(x)
    feat = feat_fg * mask_fg + feat_bg * (1 - mask_fg)
"""


class ASHEngine(nn.Module):
    """
    The core ASH engine. It maintains the minimal states of a hash map,
    namely a keys-indices map, associated with a heap.
    """

    def __init__(
        self,
        dim: int,
        capacity: int,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        """Init ASH engine.
        Args:
            dim: dimension of the keys
            capacity: maximum number of items in the hash map
            device: device to store the states
        """
        super().__init__()
        assert dim > 0, "dim must be positive"
        self.dim = dim

        assert capacity > 0, "capacity must be positive"
        self.capacity = capacity

        self.device = isinstance(device, str) and torch.device(device) or device

        self._register_buffers()
        self.backend = backend.HashMap(self.dim, self.capacity, self._heap, self.device)

    def _register_buffers(self) -> None:
        """Register buffers for state_dict save and load."""

        self.register_buffer(
            "_heap", torch.arange(self.capacity, dtype=torch.int32, device=self.device)
        )

        # DO NOT ACCESS: keys and indices are reserved for state_dict save and load.
        self.register_buffer(
            "_keys",
            torch.zeros(
                (self.capacity, self.dim), dtype=torch.int32, device=self.device
            ),
        )
        self.register_buffer(
            "_indices",
            torch.zeros(self.capacity, dtype=torch.long, device=self.device),
        )
        self.register_buffer(
            "_size", torch.zeros(1, dtype=torch.int32, device=self.device)
        )
        self.register_load_state_dict_post_hook(self._post_load_state_dict_hook)

    def state_dict(
        self, destination=None, prefix: str = "", keep_vars: bool = False
    ) -> "OrderedDict[str, torch.Tensor]":
        """Override state_dict to obtain active keys and indices into the backend.
        Args:
            destination: see torch.nn.Module.state_dict
            prefix: see torch.nn.Module.state_dict
            keep_vars: see torch.nn.Module.state_dict
        Returns:
            state_dict: state_dict with size, heap, active keys, and indices
            to reproduce the hash map.
        """
        state_dict = super().state_dict(destination, prefix, keep_vars)

        active_keys, active_indices = self.items()
        size = self.backend.size()

        self._size[:] = size
        if size > 0:
            self._keys[:size] = active_keys
            self._indices[:size] = active_indices

        return state_dict

    def _post_load_state_dict_hook(self, module, incompatible_keys) -> None:
        """hook to load active keys and indices into the backend.
        Args:
            module: see torch.nn.Module.state_dict
            incompatible_keys: see torch.nn.Module.state_dict
        """
        backend_state_keys = ["_keys", "_indices", "_size"]

        size = self._size.item()

        backend_state_dict = {
            "active_keys": self._keys[:size],
            "active_indices": self._indices[:size],
            "heap": self._heap,
        }

        assert len(self._heap) == self.capacity
        self._key_check(backend_state_dict["active_keys"])

        self.backend.load_states(backend_state_dict)

    def _key_check(self, keys: torch.Tensor) -> None:
        """Check keys shape and dtype.
        Args:
            keys: keys to check for insert, find, erase
        """
        assert len(keys.shape) == 2 and keys.shape[1] == self.dim, "keys shape mismatch"

        if keys.dtype != torch.int32:
            warnings.warn("keys are not int32, conversion might reduce precision.")

    def _value_check(
        self, values: Dict[str, torch.Tensor], external_values: Dict[str, torch.Tensor]
    ) -> None:
        """Check values shape and dtype.
        Check if insertions and external_values are consistent.
        Args:
            values: values to check for insert
            external_values: external values, usually maintained by the user as the values or embeddings
        """
        assert values.keys() == external_values.keys()
        for k, v in values.items():
            assert k in external_values

            assert v.is_contiguous()
            assert external_values[k].is_contiguous()

            assert v.ndim == external_values[k].ndim
            assert v.dtype == external_values[k].dtype
            assert v.shape[1:] == external_values[k].shape[1:]
            assert self.capacity == external_values[k].shape[0]

    def find(self, keys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find indices of keys in the hash map.
        Args:
            keys: (N, dim)
        Returns:
            indices: (N,) indices of found keys in the hash map, can be associated with maintained values
            masks: (N,) masks of whether keys are in the hash map
        """
        self._key_check(keys)

        if len(keys) == 0:
            warnings.warn("empty keys")
            return (
                torch.empty((0,), dtype=torch.long, device=self.device),
                torch.empty((0,), dtype=torch.bool, device=self.device),
            )

        keys = keys.to(self.device, torch.int32).contiguous()
        indices, masks = self.backend.find(keys)
        return indices, masks

    def insert_keys(self, keys: torch.Tensor) -> None:
        """Insert keys into the hash map without specifying values.
        Args:
            keys: (N, dim)
        """
        # TODO(wei): add safe and unsafe options
        self._key_check(keys)

        if len(keys) == 0:
            warnings.warn("empty keys")
            return

        keys = keys.to(self.device, torch.int32).contiguous()

        prev_size = self.size()
        self.backend.insert_keys(keys)
        curr_size = self.size()
        if len(keys) + prev_size > self.capacity and curr_size > self.capacity:
            warnings.warn(
                f"Insertion of {len(keys)} increased the hash map size from {prev_size} "
                f"to {curr_size}, which exceeds the hash map capacity {self.capacity}. "
                "Please resize the hash map or the behavior could be unexpected."
            )

    def insert(
        self,
        keys: torch.Tensor,
        values: Dict[str, torch.Tensor],
        external_values: Dict[str, torch.Tensor],
    ) -> None:
        """Insert keys and values into the hash map.
        Args:
            keys: (N, dim)
            values: dict of values to insert, e.g. {"value": (N, dim)}
            external_values: dict of external values, usually maintained by the user as the values or embeddings
        """
        self._key_check(keys)

        if len(keys) == 0:
            warnings.warn("empty keys")
            return

        self._value_check(values, external_values)
        for k, v in values.items():
            assert v.shape[0] == len(keys)
        keys = keys.to(self.device, torch.int32).contiguous()

        prev_size = self.size()
        self.backend.insert(keys, values, external_values)
        curr_size = self.size()
        if len(keys) + prev_size > self.capacity and curr_size > self.capacity:
            warnings.warn(
                f"Insertion of {len(keys)} increased the hash map size from {prev_size} "
                f"to {curr_size}, which exceeds the hash map capacity {self.capacity}. "
                "Please resize the hash map or the behavior could be unexpected."
            )

    def erase(self, keys: torch.Tensor) -> None:
        """Erase keys from the hash map.
        Args:
            keys: (N, dim)
        """
        self._key_check(keys)
        if len(keys) == 0:
            warnings.warn("empty keys")
            return

        self.backend.erase(keys)

    def clear(self) -> None:
        """Clear the hash map."""
        self.backend.clear()

    def size(self) -> int:
        """Return the current size of the hash map."""
        return self.backend.size()

    def resize(
        self,
        capacity: int,
        old_external_values: Dict[str, torch.Tensor] = None,
        new_external_values: Dict[str, torch.Tensor] = None,
    ) -> None:
        """Resize the hash map to a new capacity.
        Args:
            capacity: new capacity
            old_external_values: dict of external values or embeddings before resizing
            new_external_values: dict of external values or embeddings after resizing
        """
        assert capacity >= self.size(), "new capacity is smaller than the current size"
        assert capacity > 0, "new capacity is 0"

        active_keys, old_indices = self.items()
        size = self.backend.size()
        del self.backend

        self.capacity = capacity

        self._register_buffers()
        self.backend = backend.HashMap(self.dim, self.capacity, self._heap, self.device)

        if size == 0:
            return

        if old_external_values is not None and new_external_values is not None:
            active_old_external_values = {}
            for k, v in old_external_values.items():
                active_old_external_values[k] = v[old_indices]

            self.backend.insert(
                active_keys, active_old_external_values, new_external_values
            )
        else:
            self.backend.insert_keys(active_keys)

    def keys(self) -> torch.Tensor:
        """Return all active keys in the hash map."""
        active_keys, active_indices = self.items()
        return active_keys

    def items(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return all active keys and indices to external values in the hash map.
        Returns:
            active_keys: (N, dim)
            active_indices: (N,)
        """
        size = self.backend.size()
        if size == 0:
            return (
                torch.empty((0, self.dim), dtype=torch.int32, device=self.device),
                torch.empty((0,), dtype=torch.long, device=self.device),
            )
        active_keys, active_indices = self.backend.items()
        return active_keys, active_indices

    # TODO(wei): implement cpu engine and enable to and from
    def to(self, device: Union[str, torch.device]):
        device = torch.device(device) if isinstance(device, str) else device
        new_engine = ASHEngine(self.dim, self.capacity, device)

        return new_engine

    def cpu(self):
        return self.to("cpu")

    def cuda(self, device: Optional[Union[str, torch.device]] = None):
        if device is None:
            return self.to("cuda")
        return self.to(device)

    def __repr__(self) -> str:
        name = "ASHEngine (dim={}, capacity={}, size={}) at {}".format(
            self.dim,
            self.capacity,
            self.size(),
            self.device,
        )
        return name


class ASHModule(nn.Module):
    """
    Helper virtual class to handle engine's 'to' operations
    Inherited by HashEmbedding, HashMap, and HashSet
    """

    def __init__(self):
        super().__init__()
        self.engine = None

    def to(self, device):
        assert self.engine is not None, "engine is not initialized"
        module = super(ASHModule, self).to(device)
        module.engine = module.engine.to(device)
        return module

    def cpu(self):
        return self.to(torch.device("cpu"))

    def cuda(self, device: Optional[Union[str, torch.device]] = None):
        if device is None:
            return self.to(torch.device("cuda"))
        return self.to(device)
