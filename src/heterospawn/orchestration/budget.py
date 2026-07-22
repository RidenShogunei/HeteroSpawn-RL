"""Concurrency budget ledger with explicit reserve/commit/release transitions."""

from __future__ import annotations

import asyncio

from heterospawn.orchestration.models import BudgetSnapshot


class ConcurrencyBudgetLedger:
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._reserved: set[str] = set()
        self._committed: set[str] = set()
        self._lock = asyncio.Lock()

    async def reserve(self, reservation_id: str) -> None:
        async with self._lock:
            if reservation_id in self._reserved or reservation_id in self._committed:
                raise RuntimeError("duplicate budget reservation")
            if len(self._reserved) + len(self._committed) >= self._capacity:
                raise RuntimeError("concurrency budget exhausted")
            self._reserved.add(reservation_id)

    async def commit(self, reservation_id: str) -> None:
        async with self._lock:
            if reservation_id not in self._reserved:
                raise RuntimeError("cannot commit an unreserved budget slot")
            self._reserved.remove(reservation_id)
            self._committed.add(reservation_id)

    async def release(self, reservation_id: str) -> None:
        async with self._lock:
            removed = reservation_id in self._reserved or reservation_id in self._committed
            self._reserved.discard(reservation_id)
            self._committed.discard(reservation_id)
            if not removed:
                raise RuntimeError("cannot release an unknown budget slot")

    async def snapshot(self) -> BudgetSnapshot:
        async with self._lock:
            return BudgetSnapshot(
                capacity=self._capacity,
                reserved=len(self._reserved),
                committed=len(self._committed),
            )
