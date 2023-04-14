import torch
import os
from ash import HashSet, HashMap
import pytest


class TestHashSet:
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

    def _insert_block(self, dim, num):
        hashset = HashSet(dim, self.capacity, self.device)

        keys = self._generate_keys(dim, num)
        hashset.insert(keys)

        masks = hashset.find(keys)
        assert masks.sum() == num

    def _erase_block(self, dim, num):
        hashset = HashSet(dim, self.capacity, self.device)

        keys = self._generate_keys(dim, num)
        hashset.insert(keys)

        hashset.erase(keys[: num // 2])

        masks = hashset.find(keys)
        assert masks.sum() == num // 2
        assert hashset.size() == num // 2

        hashset.erase(keys)
        assert hashset.size() == 0

    def _resize_block(self, dim, num, old_capacity, new_capacity):
        hashset = HashSet(dim, old_capacity, self.device)

        keys = self._generate_keys(dim, num)
        hashset.insert(keys)

        hashset.resize(new_capacity)

        masks = hashset.find(keys)
        assert masks.sum() == num

    def _keys_block(self, dim, num):
        hashset = HashSet(dim, self.capacity, self.device)

        keys = self._generate_keys(dim, num)
        hashset.insert(keys)

        active_keys = hashset.keys()
        sorted_active_keys, _ = active_keys.sort(dim=0)
        sorted_keys, _ = keys.sort(dim=0)
        assert torch.all(sorted_active_keys == sorted_keys)

    def test_insert(self):
        self._insert_block(1, 100)
        self._insert_block(1, 1000)
        self._insert_block(1, 10000)
        self._insert_block(1, 100000)
        self._insert_block(2, 100)
        self._insert_block(2, 1000)
        self._insert_block(2, 10000)
        self._insert_block(2, 100000)
        self._insert_block(3, 100)
        self._insert_block(3, 1000)
        self._insert_block(3, 10000)
        self._insert_block(3, 100000)

    def test_erase(self):
        self._erase_block(1, 100)
        self._erase_block(1, 1000)
        self._erase_block(1, 10000)
        self._erase_block(1, 100000)
        self._erase_block(2, 100)
        self._erase_block(2, 1000)
        self._erase_block(2, 10000)
        self._erase_block(2, 100000)
        self._erase_block(3, 100)
        self._erase_block(3, 1000)
        self._erase_block(3, 10000)
        self._erase_block(3, 100000)

    def test_resize(self):
        self._resize_block(1, 100, 100, 1000)
        self._resize_block(1, 1000, 1000, 10000)
        self._resize_block(1, 10000, 10000, 100000)
        self._resize_block(1, 100000, 100000, 1000000)
        self._resize_block(2, 100, 100, 1000)
        self._resize_block(2, 1000, 1000, 10000)
        self._resize_block(2, 10000, 10000, 100000)
        self._resize_block(2, 100000, 100000, 1000000)
        self._resize_block(3, 100, 100, 1000)
        self._resize_block(3, 1000, 1000, 10000)
        self._resize_block(3, 10000, 10000, 100000)
        self._resize_block(3, 100000, 100000, 1000000)

    def test_keys(self):
        self._keys_block(1, 100)
        self._keys_block(1, 1000)
        self._keys_block(1, 10000)
        self._keys_block(1, 100000)
        self._keys_block(2, 100)
        self._keys_block(2, 1000)
        self._keys_block(2, 10000)
        self._keys_block(2, 100000)
        self._keys_block(3, 100)
        self._keys_block(3, 1000)
        self._keys_block(3, 10000)
        self._keys_block(3, 100000)
