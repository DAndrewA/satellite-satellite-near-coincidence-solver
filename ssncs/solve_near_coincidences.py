"""Script for solving for near-coincidences between two satellite subpoint trajectories, within a time frame given by the user."""


# 1 load TLEs for both satellites
# 2 create array of datetime objects with sufficient density to allow detection of trajectory crossings
# 3 compute subpoint positions at all times
# 4 create ring-buffer of s2 position PathSegments that contains just enough path segements to cover the ±tau window centered on the first lat-lon coords from s1
# 5 for each adjacent pair of lat-lon coords from s1: 
    # 5.1 calculate the PathSegment for s1
    # 5.2 assess intersections between the s1 path segement and all PathSegments in the s2 ring buffer
    # 5.3 if there's an intersection, save it :)
    # 5.4 Push the next PathSegment to s2 ring buffer
    # 5.4 progress to the next PathSegment from s1

from . import load_tle
from .ring import FIFORingBuffer as RingBuffer
from .trajectory import get_datetimes
from .multi_tle import MultiTLE
from .path_segment import PathSegment

from skyfield.api import Timescale, wgs84
import datetime as dt
import numpy as np


def get_datetimes(start: dt.datetime, end: dt.datetime, freq: dt.timedelta = dt.timedelta(seconds=60), tau: dt.timedelta = dt.timedelta(minutes=20)) -> np.ndarray:
    """Function taking a start and end utc datetime, and create a set of datetimes with fixed frequency, and convert to a skyfield Time array"""
    datetimes = pd.date_range(
        start=start,
        end=end,
        freq=freq,
        tz=dt.timezone.utc,
    ).to_pydatetime()
    return datetimes#ts.from_datetimes(datetimes)

def solve_near_coincidences(sat_1: MultiTLE, sat_2: MultiTLE, start: dt.datetime, end:dt.datetime, ts: Timescale):
    datetimes = get_datetimes(start, end, freq="60s")

    xy1 = np.stack( sat_1.get_lon_lat_at_datetimes(datetimes, ts), axis=1)
    xy2 = np.stack( sat_2.get_lon_lat_at_datetimes(datetimes, ts), axis=1)

    N_halfwidth = int(tau/freq) + 1
    N = 2*N + 1
    
    # create a ring buffer of length N and insert xy2 positions into it to fill it
    ring_buffer = RingBuffer(capcity=N-1)
    ring_buffer.push([
        PathSegment(start=xy21, end=xy22)
        for xy21, xy22 in zip( xy2[:N-1,:], xy2[1:N,:] )
    ])
    xy2 = xy2[N-1:]
    xy21 = xy2[0,:]
    xy22 = xy2[1,:]
    xy1 = xy1[N_halfwidth:-N_halfwidth]

    for i, (xy11, dtime) in enumerate(zip(xy1,datetimes[N_halfwidth:-N_halfwidth])):
        xy12 = xy1[i+1,:]
        ps1 = PathSegment(start=xy11, end=xy12)

        if np.any([ps1.intersects(ps2) for ps2 in ring_buffer.see()]):
            raise NotImplementedError(f"Intersection found for {i=}, {xy11=}, {dtime=}")

        xy2 = xy2[1:]
        xy21 = xy22
        xy22 = xy2[1]
        



    
    
