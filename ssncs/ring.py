"""FIFORingBuffer class definition for use in itterating through datetimes, positions, and PathSegments dfefined along TLE-propogated trajectories"""

from __future__ import annotations
from typing import Any

class FIFORingBuffer:
    def __init__(self, capacity):
        self.buffer = [None] * capacity
        self.capacity = capacity
        self.fi = 0
        self.elements = 0

    def increment_fi(self, n: int=1):
        self.fi = self.get_fi_displacement(n)

    def get_fi_displacement(self, n: int) -> int:
        return (self.fi + n)%self.capacity

    def push(self, *obs):
        for ob in obs:
            self.push_single(ob)

    def push_single(self, ob):
        insertion_index = self.get_fi_displacement(self.elements)
        self.buffer[insertion_index] = ob
        if insertion_index == self.fi:
            self.increment_fi()
        else:
            self.elements += 1

    def pop(self, n: int = 1) -> list[Any]:
        poplist = []
        for _ in range(n):
            popped = self.pop_single()
            if popped is not None:
                poplist.append(popped)
        return poplist

    def pop_single(self) -> Any:
        ob = self.buffer[self.fi]
        self.buffer[self.fi] = None
        self.increment_fi()
        self.elements -= 1
        return ob

    def see_all(self) -> list[Any | None]:
        return self.buffer[self.fi:] + self.buffer[:self.fi]

    def see(self) -> list[Any]:
        return [ob for ob in self.see_all() if ob is not None]

