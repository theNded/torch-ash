import torch
import os
from ash import ASHEngine
import pytest


class TestEngine:
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

    def _insert_find_block(self, dim, num):
        engine = ASHEngine(dim=dim, capacity=self.capacity, device=self.device)

        keys = self._generate_keys(dim=dim, num=num)

        engine.insert_keys(keys)
        assert engine.size() == num

        indices, masks = engine.find(keys)
        assert masks.sum() == num

        unique_indices = torch.unique(indices, sorted=True)
        assert len(unique_indices) == self.capacity
        assert unique_indices.eq(
            torch.arange(num, dtype=torch.int32, device=self.device)
        ).all()

    def _insert_erase_block(self, dim, num):
        engine = ASHEngine(dim=dim, capacity=self.capacity, device=self.device)

        keys = self._generate_keys(dim=dim, num=num)

        engine.insert_keys(keys)
        assert engine.size() == num

        engine.erase(keys)
        assert engine.size() == 0

    def _insert_duplicate_block(self, dim, num):
        engine = ASHEngine(dim=dim, capacity=self.capacity, device=self.device)

        # duplicate 3 times. can be inserted once but found twice
        keys = self._generate_keys(dim=dim, num=num)
        dup_keys = keys.tile((3, 1))
        perm = torch.randperm(len(dup_keys), device=self.device)
        dup_keys = dup_keys[perm]

        engine.insert_keys(dup_keys)
        assert engine.size() == num

        # could be found all the times
        indices, masks = engine.find(dup_keys)
        assert masks.sum() == num * 3

        unique_indices = torch.unique(indices, sorted=True)
        assert len(unique_indices) == num
        assert unique_indices.eq(
            torch.arange(num, dtype=torch.int32, device=self.device)
        ).all()

    def _items_block(self, dim, num):
        engine = ASHEngine(dim=dim, capacity=self.capacity, device=self.device)

        keys = self._generate_keys(dim=dim, num=num)

        engine.insert_keys(keys)
        assert engine.size() == num

        active_keys, active_indices = engine.items()
        sorted_inserted_keys, _ = torch.sort(keys, dim=0)
        sorted_active_keys, _ = torch.sort(active_keys, dim=0)

        assert sorted_active_keys.eq(sorted_inserted_keys).all()

    def _state_dict_block(self, dim, num):
        engine = ASHEngine(dim=dim, capacity=self.capacity, device=self.device)

        keys = self._generate_keys(dim=dim, num=num)
        engine.insert_keys(keys)

        state_dict = engine.state_dict()
        torch.save(state_dict, "test_state_dict.pt")
        del engine

        state_dict = torch.load("test_state_dict.pt", map_location=self.device)
        os.remove("test_state_dict.pt")

        engine = ASHEngine(dim=dim, capacity=self.capacity, device=self.device)
        engine.load_state_dict(state_dict)

        active_keys, active_indices = engine.items()
        assert len(active_keys) == num
        sorted_inserted_keys, _ = torch.sort(keys, dim=0)
        sorted_active_keys, _ = torch.sort(active_keys, dim=0)

        assert sorted_active_keys.eq(sorted_inserted_keys).all()

    def _resize_block(self, dim, key_nums, old_capacity, new_capacity):
        engine = ASHEngine(dim=dim, capacity=old_capacity, device=self.device)
        keys = self._generate_keys(dim=dim, num=key_nums)

        engine.insert_keys(keys)
        assert engine.size() == key_nums

        engine.resize(new_capacity)
        active_keys, active_indices = engine.items()
        assert len(active_keys) == key_nums
        sorted_inserted_keys, _ = torch.sort(keys, dim=0)
        sorted_active_keys, _ = torch.sort(active_keys, dim=0)

    def _multivalue_block(self, dim, num, old_capacity, new_capacity):
        engine = ASHEngine(dim=dim, capacity=old_capacity, device=self.device)
        old_embeddings = {
            "0": torch.zeros(old_capacity, dtype=torch.int64, device=self.device)
            .view(-1, 1)
            .tile((1, dim)),
            "1": torch.zeros(old_capacity, dtype=torch.float32, device=self.device)
            .view(-1, 1)
            .tile((1, dim)),
        }

        keys = self._generate_keys(dim=dim, num=num)
        values = {
            "0": keys.clone().long() * 2,
            "1": keys.clone().float() * 4,
        }

        # Insert/find multi-keys
        engine.insert(keys, values, old_embeddings)
        indices, masks = engine.find(keys)
        assert masks.sum() == num
        assert torch.eq(old_embeddings["0"][indices], keys.long() * 2).all()
        assert torch.allclose(old_embeddings["1"][indices], keys.float() * 4)

        # Resize multi-keys
        new_embeddings = {
            "0": torch.zeros(new_capacity, dtype=torch.int64, device=self.device)
            .view(-1, 1)
            .tile((1, dim)),
            "1": torch.zeros(new_capacity, dtype=torch.float32, device=self.device)
            .view(-1, 1)
            .tile((1, dim)),
        }
        engine.resize(new_capacity, old_embeddings, new_embeddings)

        indices, masks = engine.find(keys)
        assert masks.sum() == num
        assert torch.eq(new_embeddings["0"][indices], keys.long() * 2).all()
        assert torch.allclose(new_embeddings["1"][indices], keys.float() * 4)

    # Basics
    def test_insert_find(self):
        self._insert_find_block(dim=1, num=self.capacity)
        self._insert_find_block(dim=2, num=self.capacity)
        self._insert_find_block(dim=3, num=self.capacity)
        self._insert_find_block(dim=4, num=self.capacity)

    def test_insert_erase(self):
        self._insert_erase_block(dim=1, num=self.capacity)
        self._insert_erase_block(dim=2, num=self.capacity)
        self._insert_erase_block(dim=3, num=self.capacity)
        self._insert_erase_block(dim=4, num=self.capacity)

    @pytest.mark.filterwarnings("ignore:insertions")
    def test_insert_duplicate(self):
        self._insert_duplicate_block(dim=1, num=self.capacity)
        self._insert_duplicate_block(dim=2, num=self.capacity)
        self._insert_duplicate_block(dim=3, num=self.capacity)
        self._insert_duplicate_block(dim=4, num=self.capacity)

    # States
    def test_items_1d(self):
        self._items_block(dim=1, num=self.capacity)
        self._items_block(dim=2, num=self.capacity)
        self._items_block(dim=3, num=self.capacity)
        self._items_block(dim=4, num=self.capacity)

    @pytest.mark.filterwarnings("ignore:empty")
    def test_state_dict(self):
        self._state_dict_block(dim=1, num=0)
        self._state_dict_block(dim=2, num=0)
        self._state_dict_block(dim=3, num=0)
        self._state_dict_block(dim=4, num=0)

        self._state_dict_block(dim=1, num=self.capacity)
        self._state_dict_block(dim=2, num=self.capacity)
        self._state_dict_block(dim=3, num=self.capacity)
        self._state_dict_block(dim=4, num=self.capacity)

    def test_resize(self):
        self._resize_block(
            dim=1,
            key_nums=self.capacity // 2,
            old_capacity=self.capacity,
            new_capacity=self.capacity // 2,
        )
        self._resize_block(
            dim=1,
            key_nums=self.capacity,
            old_capacity=self.capacity,
            new_capacity=self.capacity * 2,
        )
        self._resize_block(
            dim=3,
            key_nums=self.capacity // 2,
            old_capacity=self.capacity,
            new_capacity=self.capacity // 2,
        )
        self._resize_block(
            dim=3,
            key_nums=self.capacity,
            old_capacity=self.capacity,
            new_capacity=self.capacity * 2,
        )

    def test_multivalue(self):
        self._multivalue_block(
            dim=1,
            num=self.capacity // 2,
            old_capacity=self.capacity,
            new_capacity=self.capacity // 2,
        )
        self._multivalue_block(
            dim=1,
            num=self.capacity,
            old_capacity=self.capacity,
            new_capacity=self.capacity * 2,
        )
        self._multivalue_block(
            dim=3,
            num=self.capacity // 2,
            old_capacity=self.capacity,
            new_capacity=self.capacity // 2,
        )
        self._multivalue_block(
            dim=3,
            num=self.capacity,
            old_capacity=self.capacity,
            new_capacity=self.capacity * 2,
        )
