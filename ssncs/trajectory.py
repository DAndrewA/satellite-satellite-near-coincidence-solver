"""Functions and class definitions for obtaining lat-lon trajectories from propogated TLEs, and assessing if they intersect within certain temporal bounds"""

from .path_segment import PathSegment
from .ring import FIFORingBuffer
from skyfield.api import load, EarthSatellite, Timescale, Time
import datetime as dt
import numpy as np
import pandas as pd
from typing import Any


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
