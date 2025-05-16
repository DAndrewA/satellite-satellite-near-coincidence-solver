"""Functions and class definitions for obtaining lat-lon trajectories from propogated TLEs, and assessing if they intersect within certain temporal bounds"""

from .path_segment import PathSegment
from skyfield.api import load, EarthSatellite, Timescale, Time
import datetime as dt
import numpy as np
import pandas as pd
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



def get_datetimes(start: dt.datetime, end: dt.datetime, ts: Timescale, freq: str="60s") -> Time:
    """Function taking a start and end utc datetime, and create a set of datetimes with fixed frequency, and convert to a skyfield Time array"""
    datetimes = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        tz=dt.timezone.utc,
    ).to_pydatetime()
    return ts.from_datetimes(datetimes)

def propogate_satellite(satellite: EarthSatellite, times: Time) -> EarthSatellite:
    return satellite.at(times)


# function to get lat-lon from EarthSatellite subpoint, as arrays
# function to 
