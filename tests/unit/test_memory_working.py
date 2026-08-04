from __future__ import annotations

import pytest

from agent.memory.working import WorkingMemory


class TestInit:
    def test_starts_empty(self) -> None:
        m = WorkingMemory()
        assert len(m) == 0

    def test_capacity_stored(self) -> None:
        m = WorkingMemory(max_entries=42)
        assert m.capacity == 42

    def test_rejects_zero_capacity(self) -> None:
        with pytest.raises(ValueError):
            WorkingMemory(max_entries=0)

    def test_rejects_negative_capacity(self) -> None:
        with pytest.raises(ValueError):
            WorkingMemory(max_entries=-1)


class TestPut:
    def test_put_increments_length(self) -> None:
        m = WorkingMemory(max_entries=10)
        m.put("k", "v")
        assert len(m) == 1

    def test_evicts_oldest_when_full(self) -> None:
        m = WorkingMemory(max_entries=3)
        m.put("a", 1)
        m.put("b", 2)
        m.put("c", 3)
        m.put("d", 4)  # should evict ("a", 1)
        assert len(m) == 3
        keys = [k for k, _ in m.get_all()]
        assert "a" not in keys
        assert "d" in keys

    def test_evicts_in_fifo_order(self) -> None:
        m = WorkingMemory(max_entries=2)
        m.put("first", 1)
        m.put("second", 2)
        m.put("third", 3)
        entries = m.get_all()
        assert entries[0] == ("second", 2)
        assert entries[1] == ("third", 3)

    def test_accepts_any_value_type(self) -> None:
        m = WorkingMemory()
        m.put("list", [1, 2, 3])
        m.put("dict", {"x": 1})
        m.put("none", None)
        assert len(m) == 3


class TestGetAll:
    def test_returns_in_insertion_order(self) -> None:
        m = WorkingMemory()
        m.put("x", 10)
        m.put("y", 20)
        m.put("z", 30)
        assert m.get_all() == [("x", 10), ("y", 20), ("z", 30)]

    def test_returns_empty_list_when_empty(self) -> None:
        assert WorkingMemory().get_all() == []

    def test_returns_copy_not_reference(self) -> None:
        m = WorkingMemory()
        m.put("k", "v")
        snapshot = m.get_all()
        m.put("k2", "v2")
        assert len(snapshot) == 1  # original snapshot unchanged


class TestGetByKey:
    def test_returns_all_values_for_key(self) -> None:
        m = WorkingMemory()
        m.put("ctx", "first")
        m.put("other", "x")
        m.put("ctx", "second")
        assert m.get_by_key("ctx") == ["first", "second"]

    def test_returns_empty_for_missing_key(self) -> None:
        m = WorkingMemory()
        assert m.get_by_key("missing") == []


class TestLatest:
    def test_returns_most_recent_for_key(self) -> None:
        m = WorkingMemory()
        m.put("k", "old")
        m.put("k", "new")
        assert m.latest("k") == "new"

    def test_returns_none_for_missing_key(self) -> None:
        assert WorkingMemory().latest("nope") is None


class TestClear:
    def test_clear_empties_buffer(self) -> None:
        m = WorkingMemory()
        m.put("a", 1)
        m.put("b", 2)
        m.clear()
        assert len(m) == 0
        assert m.get_all() == []

    def test_can_add_after_clear(self) -> None:
        m = WorkingMemory(max_entries=2)
        m.put("a", 1)
        m.put("b", 2)
        m.clear()
        m.put("c", 3)
        assert len(m) == 1


class TestIsFull:
    def test_not_full_when_empty(self) -> None:
        assert not WorkingMemory(max_entries=5).is_full

    def test_full_at_capacity(self) -> None:
        m = WorkingMemory(max_entries=2)
        m.put("a", 1)
        m.put("b", 2)
        assert m.is_full

    def test_still_full_after_eviction(self) -> None:
        m = WorkingMemory(max_entries=2)
        for i in range(5):
            m.put("k", i)
        assert m.is_full
