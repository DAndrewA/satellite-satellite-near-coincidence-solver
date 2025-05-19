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
from .path_segment import PathSegment, Intersection

from skyfield.api import Timescale, wgs84
import datetime as dt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs


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
        



class SatelliteSatelliteNearCoincidenceSolver:
    def __init__(self, tau: dt.timedelta, freq: dt.timedelta, ts: Timescale, sat1: MultiTLE, sat2: MultiTLE):
        assert isinstance(tau, dt.timedelta), f"{type(tau)=} should be instance of datetime.timedelta"
        assert isinstance(freq, dt.timedelta), f"{type(freq)=} should be instance of datetime.timedelta"
        assert isinstance(ts, Timescale), f"{type(ts)=} should be instance of Timescale"
        assert isinstance(sat1, MultiTLE), f"{type(sat1)=} should be instance of MultiTLE"
        assert isinstance(sat2, MultiTLE), f"{type(sat2)=} should be instance of MultiTLE"
        
        self.tau = tau
        self.freq = freq
        self.ts = ts
        self.sat1 = sat1
        self.sat2 = sat2

        self.N_halfwidth = int(tau / freq) + 1
        self.N = 2*self.N_halfwidth + 1
    
    def from_datetime_range(self, start=dt.datetime, end=dt.datetime):
        datetimes = self._create_datetime_range(start, end)

        s1_lons_lats = np.stack(self.sat1.get_lon_lat_at_datetimes(datetimes, self.ts), axis=1)
        s2_lons_lats = np.stack(self.sat2.get_lon_lat_at_datetimes(datetimes, self.ts), axis=1)
        
        return RingBufferSolver(
            datetimes=datetimes,
            s1_lons_lats=s1_lons_lats,
            s2_lons_lats=s2_lons_lats,
            N=self.N,
            N_half=self.N_halfwidth
        )


    def from_datetime_range_with_lerp(self, start=dt.datetime, end=dt.datetime):
        datetimes = self._create_datetime_range(start, end)

        s1_lons_lats = np.stack(self.sat1.get_lon_lat_at_datetimes_with_lerp(datetimes, self.ts), axis=1)
        s2_lons_lats = np.stack(self.sat2.get_lon_lat_at_datetimes_with_lerp(datetimes, self.ts), axis=1)
        
        return RingBufferSolver(
            datetimes=datetimes,
            s1_lons_lats=s1_lons_lats,
            s2_lons_lats=s2_lons_lats,
            N=self.N,
            N_half=self.N_halfwidth
        )


    def _create_datetime_range(self, start=dt.datetime, end=dt.datetime):
        datetimes = pd.date_range(
            start = start,
            end = end,
            freq=self.freq,
            tz = dt.timezone.utc
        ).to_pydatetime()
        return datetimes


class RingBufferSolver:
    def __init__(self, datetimes: np.ndarray, s1_lons_lats: np.ndarray, s2_lons_lats: np.ndarray, N: int, N_half: int):
        M = datetimes.size
        assert s1_lons_lats.shape == (M,2), f"{s1_lons_lats.shape=} must be ({M}, 2)"
        assert s2_lons_lats.shape == (M,2), f"{s2_lons_lats.shape=} must be ({M}, 2)"
        
        assert type(N) == int, f"{type(N)=} should be int"
        assert type(N_half) == int, f"{type(N_half)=} should be int"
        assert 2*N_half + 1 == N, f"{2*N_half+1=} should equal {N=}"

        self.datetimes = datetimes
        self.s1_lons_lats = s1_lons_lats
        self.s2_lons_lats = s2_lons_lats
        self.N = N
        self.N_halfwidth = N_half

    def solve(self):
        self.intersections = list()
        plate_carree = ccrs.PlateCarree()
        # initialise ring buffer with self.N-1 Path Segments representing ±tau from central point
        ring = RingBuffer(capacity = self.N - 1) 
        #ring.push(*[
        #    PathSegment(
        #        start = xy21,
        #        end = xy22,
        #        crs = plate_carree,
        #        meta = dtime
        #    )
        #    for dtime, xy21, xy22 in zip(
        #        self.datetimes[:self.N-1],
        #        self.s2_lons_lats[:self.N-1, :],
        #        self.s2_lons_lats[1:self.N, :]
        #    )
        #])
        # for loop not inlined to preserve xy21, xy22 variable names
        for dtime2, xy21, xy22 in zip(
            self.datetimes[:self.N-1],
            self.s2_lons_lats[:self.N-1, :],
            self.s2_lons_lats[1:self.N, :]
        ):
            ring.push(
                PathSegment(
                    start = xy21,
                    end = xy22,
                    crs = plate_carree,
                    meta = dtime2
                )
            )

        [xy11, xy12] = self.s1_lons_lats[self.N_halfwidth: self.N_halfwidth+2 , :]

        for dtime1, xy11, xy12, new_xy22, dtime2 in zip(
            self.datetimes[self.N_halfwidth:],
            self.s1_lons_lats[self.N_halfwidth:, :],
            self.s1_lons_lats[self.N_halfwidth+1:, :],
            self.s2_lons_lats[self.N:, :],
            self.datetimes[self.N-1:]
        ):
            ps1 = PathSegment(
                start = xy11,
                end = xy12,
                crs = plate_carree,
                meta = dtime1
            )
            crs_ps1 = ccrs.AzimuthalEquidistant(central_longitude=xy11[0], central_latitude=xy11[1])

            ps1_transformed = ps1.transform_crs(crs_ps1)
            for ps2 in ring.see():
                if ps1.intersects( ps2.transform_crs(crs_ps1) ):
                    self.intersections.append(
                        Intersection.from_intersecting_path_segments( ps1, ps2 )
                    )

            # push new sat2 PathSegment to ring
            xy21 = xy22
            xy22 = new_xy22
            ring.push(
                PathSegment(
                    start=xy21,
                    end=xy22,
                    crs=plate_carree,
                    meta=dtime2
                )
            )




