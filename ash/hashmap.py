from typing import List, Union, Tuple, Dict

import torch
from .core import ASHEngine, ASHModule


class HashMap(ASHModule):
    """
    Standalone hash map powered by ASHEngine, with maintained values.
    """

    def __init__(
        self,
        key_dim: int,
        value_dims: Union[int, List[int], Dict[str, int]],
        capacity: int,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        self.engine = ASHEngine(key_dim, capacity, device)
        self.value_dims = value_dims

        if isinstance(value_dims, int):
            self.values = {"default": torch.zeros(capacity, value_dims, device=device)}
        elif isinstance(value_dims, list):
            self.values = {
                f"default_{i}": torch.zeros(capacity, d, device=device)
                for i, d in enumerate(value_dims)
            }
        elif isinstance(value_dims, dict):
            self.values = {
                k: torch.zeros(capacity, d, device=device)
                for k, d in value_dims.items()
            }
        else:
            raise ValueError("value_dims must be int, list or dict")

        for k, v in self.values.items():
            self.register_buffer(k, v)

    def insert(
        self,
        keys: torch.Tensor,
        values: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
    ):
        if isinstance(values, torch.Tensor):
            assert isinstance(self.value_dims, int)
            values = {"default": values}
            self.engine.insert(keys, values, self.values)
        elif isinstance(values, list):
            assert isinstance(self.value_dims, list)
            values = {f"default_{i}": v for i, v in enumerate(values)}
            self.engine.insert(keys, values, self.values)
        elif isinstance(values, dict):
            assert isinstance(self.value_dims, dict)
            self.engine.insert(keys, values, self.values)
        else:
            raise ValueError("values must be Tensor, list, or dict")

    def reinterpret_value_dict(
        self, values: Dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(self.value_dims, int):
            return values["default"]
        elif isinstance(self.value_dims, list):
            return [values[f"default_{i}"] for i in range(len(self.value_dims))]
        elif isinstance(self.value_dims, dict):
            return values
        else:
            raise ValueError("values must be Tensor, list, or dict")

    def find(
        self,
        keys: torch.Tensor,
        return_indices: bool = True,
        return_values: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        indices, masks = self.engine.find(keys)

        assert (
            return_indices or return_values
        ), "Must return at least one of indices or values"
        ans = (masks,)
        if return_indices:
            ans = (*ans, indices)
        if return_values:
            values = {k: v[indices] for k, v in self.values.items()}
            values = self.reinterpret_value_dict(values)

            ans = (*ans, values)
        return ans

    def erase(self, keys: torch.Tensor) -> None:
        self.engine.erase(keys)

    def clear(self) -> None:
        for k, v in self.values.items():
            v.zero_()
        self.engine.clear()

    def size(self) -> int:
        return self.engine.size()

    def resize(self, new_capacity: int) -> None:
        new_values = {
            k: torch.zeros(new_capacity, *v.shape[1:], device=self.engine.device)
            for k, v in self.values.items()
        }
        self.engine.resize(new_capacity, self.values, new_values)

        # De-register old buffers
        for k, v in self.values.items():
            delattr(self, k)

        # Re-register new buffers
        self.values = new_values
        for k, v in self.values.items():
            self.register_buffer(k, v)

    def keys(self) -> torch.Tensor:
        return self.engine.keys()

    def items(
        self, return_indices: bool = True, return_values: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        assert (
            return_indices or return_values
        ), "Must return at least one of indices or values"

        active_keys, active_indices = self.engine.items()
        ans = (active_keys,)
        if return_indices:
            ans = (*ans, active_indices)
        if return_values:
            values = {k: v[active_indices] for k, v in self.values.items()}
            values = self.reinterpret_value_dict(values)
            ans = (*ans, values)
        return ans


class HashSet(ASHModule):
    """
    Standalone hash set powered by ASHEngine, with no values.
    """

    def __init__(self, key_dim: int, capacity: int, device: torch.device):
        super().__init__()
        self.engine = ASHEngine(key_dim, capacity, device)

    def insert(self, keys: torch.Tensor) -> None:
        self.engine.insert_keys(keys)

    def find(self, keys: torch.Tensor) -> torch.Tensor:
        indices, masks = self.engine.find(keys)
        return masks

    def erase(self, keys: torch.Tensor) -> None:
        self.engine.erase(keys)

    def clear(self) -> None:
        self.engine.clear()

    def size(self) -> int:
        return self.engine.size()

    def resize(self, new_capacity: int) -> None:
        self.engine.resize(new_capacity)

    def keys(self) -> torch.Tensor:
        return self.engine.keys()
