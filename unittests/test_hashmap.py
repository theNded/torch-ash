import torch
from ash import HashMap


class TestMap:
    capacity = 100000
    device = torch.device("cuda:0")

    def _generate_keys(self, dim, num):
        keys = (
            torch.arange(num, dtype=torch.int32, device=self.device)
            .view(-1, 1)
            .tile((1, dim))
        )
        perm = torch.randperm(len(keys), device=self.device)
        keys = keys[perm]
        return keys

    def _init_block(self, dim, value_dims):
        hashmap = HashMap(dim, value_dims, self.capacity, self.device)

    def _insert_find_single_value_block(self, dim, value_dim, num):
        hashmap = HashMap(dim, value_dim, self.capacity, self.device)

        keys = self._generate_keys(dim, num)

        values = torch.rand(num, value_dim, device=self.device)

        hashmap.insert(keys, values)

        masks, indices, found_values = hashmap.find(
            keys, return_indices=True, return_values=True
        )
        assert masks.sum() == num

        assert torch.allclose(found_values, hashmap.values["default"][indices])

    def _insert_find_value_list_block(self, dim, value_dims, num):
        hashmap = HashMap(dim, value_dims, self.capacity, self.device)

        keys = self._generate_keys(dim, num)

        values = [
            torch.rand(num, value_dim, device=self.device) for value_dim in value_dims
        ]

        hashmap.insert(keys, values)

        masks, indices, found_values = hashmap.find(
            keys, return_indices=True, return_values=True
        )
        assert masks.sum() == num

        for i, value_dim in enumerate(value_dims):
            assert torch.allclose(
                found_values[i], hashmap.values[f"default_{i}"][indices]
            )

    def _insert_find_value_dict_block(self, dim, value_dims, num):
        hashmap = HashMap(dim, value_dims, self.capacity, self.device)

        keys = self._generate_keys(dim, num)

        values = {
            key: torch.rand(num, value_dim, device=self.device)
            for key, value_dim in value_dims.items()
        }

        hashmap.insert(keys, values)

        masks, indices, found_values = hashmap.find(
            keys, return_indices=True, return_values=True
        )
        assert masks.sum() == num

        for key, value_dim in value_dims.items():
            assert torch.allclose(found_values[key], hashmap.values[key][indices])

    def _items_single_value_block(self, dim, value_dim, num):
        hashmap = HashMap(dim, value_dim, self.capacity, self.device)

        keys = self._generate_keys(dim, num)

        values = torch.rand(num, value_dim, device=self.device)

        hashmap.insert(keys, values)

        active_keys, active_indices, active_values = hashmap.items(
            return_indices=True, return_values=True
        )
        assert len(active_keys) == num
        sorted_active_keys, _ = torch.sort(active_keys, dim=0)
        sorted_keys, _ = torch.sort(keys, dim=0)
        assert torch.all(sorted_active_keys == sorted_keys)

        assert torch.allclose(active_values, hashmap.values["default"][active_indices])

    def _items_value_list_block(self, dim, value_dims, num):
        hashmap = HashMap(dim, value_dims, self.capacity, self.device)

        keys = self._generate_keys(dim, num)

        values = [
            torch.rand(num, value_dim, device=self.device) for value_dim in value_dims
        ]

        hashmap.insert(keys, values)

        active_keys, active_indices, active_values = hashmap.items(
            return_indices=True, return_values=True
        )
        assert len(active_keys) == num
        sorted_active_keys, _ = torch.sort(active_keys, dim=0)
        sorted_keys, _ = torch.sort(keys, dim=0)
        assert torch.all(sorted_active_keys == sorted_keys)

        for i, value_dim in enumerate(value_dims):
            assert torch.allclose(
                active_values[i], hashmap.values[f"default_{i}"][active_indices]
            )

    def _items_value_dict_block(self, dim, value_dims, num):
        hashmap = HashMap(dim, value_dims, self.capacity, self.device)

        keys = self._generate_keys(dim, num)

        values = {
            key: torch.rand(num, value_dim, device=self.device)
            for key, value_dim in value_dims.items()
        }

        hashmap.insert(keys, values)

        active_keys, active_indices, active_values = hashmap.items(
            return_indices=True, return_values=True
        )
        assert len(active_keys) == num
        sorted_active_keys, _ = torch.sort(active_keys, dim=0)
        sorted_keys, _ = torch.sort(keys, dim=0)
        assert torch.all(sorted_active_keys == sorted_keys)

        for key, value_dim in value_dims.items():
            assert torch.allclose(
                active_values[key], hashmap.values[key][active_indices]
            )

    def _resize_value_dict_block(self, dim, value_dims, num):
        hashmap = HashMap(dim, value_dims, self.capacity, self.device)

        keys = self._generate_keys(dim, num)

        values = {
            key: torch.rand(num, value_dim, device=self.device)
            for key, value_dim in value_dims.items()
        }

        hashmap.insert(keys, values)
        hashmap.resize(self.capacity * 2)

        active_keys, active_indices, active_values = hashmap.items(
            return_indices=True, return_values=True
        )
        assert len(active_keys) == num
        sorted_active_keys, _ = torch.sort(active_keys, dim=0)
        sorted_keys, _ = torch.sort(keys, dim=0)
        assert torch.all(sorted_active_keys == sorted_keys)

        for key, value_dim in value_dims.items():
            assert torch.allclose(
                active_values[key], hashmap.values[key][active_indices]
            )

    def _state_dict_block(self, dim, value_dims, num):
        hashmap = HashMap(dim, value_dims, self.capacity, self.device)

        keys = self._generate_keys(dim, num)

        values = {
            key: torch.rand(num, value_dim, device=self.device)
            for key, value_dim in value_dims.items()
        }

        hashmap.insert(keys, values)
        state_dict = hashmap.state_dict()
        torch.save(state_dict, "test_state_dict.pt")
        del hashmap
        del state_dict

        state_dict = torch.load("test_state_dict.pt", map_location=self.device)
        hashmap = HashMap(dim, value_dims, self.capacity, self.device)
        hashmap.load_state_dict(state_dict, strict=False)

        active_keys, active_indices, active_values = hashmap.items(
            return_indices=True, return_values=True
        )
        assert len(active_keys) == num
        sorted_active_keys, _ = torch.sort(active_keys, dim=0)
        sorted_keys, _ = torch.sort(keys, dim=0)
        assert torch.all(sorted_active_keys == sorted_keys)

        for key, value_dim in value_dims.items():
            assert torch.allclose(
                active_values[key], hashmap.values[key][active_indices]
            )

    def test_init(self):
        self._init_block(1, 1)
        self._init_block(2, [1, 2, 3])
        self._init_block(3, {"a": 4, "b": 10, "c": 1})

    def test_insert(self):
        self._insert_find_single_value_block(1, 1, self.capacity)
        self._insert_find_value_list_block(2, [1, 2, 3], self.capacity)
        self._insert_find_value_dict_block(3, {"a": 4, "b": 10, "c": 1}, self.capacity)

    def test_item(self):
        self._items_single_value_block(1, 1, self.capacity)
        self._items_value_list_block(2, [1, 2, 3], self.capacity)
        self._items_value_dict_block(3, {"a": 4, "b": 10, "c": 1}, self.capacity)

    def test_resize(self):
        self._resize_value_dict_block(3, {"a": 4, "b": 10, "c": 1}, self.capacity)

    def test_state_dict(self):
        self._state_dict_block(3, {"a": 4, "b": 10, "c": 1}, self.capacity)
