"""Class definition for handling multiple TLEs with different epochs, and generating positions, etc, across large timescales"""

from skyfield.api import EarthSatellite, Timescale, wgs84
from typing import Self
import datetime as dt
import numpy as np
import os
import json

class MultiTLE:
    def __init__(self, tles: list[EarthSatellite]):
        for es in tles:
            assert isinstance(es, EarthSatellite), f"type(tles[{i}])={type(es)} should be an instance of EarthSatellite."

        epochs = np.array([
            #dt.datetime.fromisoformat(es.epoch).replace(tzinfo=dt.timezone.utc)
            es.epoch.astimezone(dt.timezone.utc)
            for es in tles
        ])
        self.tles = tles
        self.epochs = epochs


    def propogate_to_datetimes(self, datetimes: np.ndarray, ts: Timescale):
        optimum_tle_for_datetime = np.argmin(
            np.abs( np.expand_dims(datetimes, -1) - self.epochs ),
            axis=-1
        )
        datetimes_for_tle = [
            datetimes[optimum_tle_for_datetime == i]
            for i in range(len(self.epochs))
        ]
        
        positions_at_datetimes = [
            es.at(ts.from_datetimes(dts_for_tle))
            for es, dts_for_tle in zip(self.tles, datetimes_for_tle)
        ]

        return positions_at_datetimes

    def get_lon_lat_at_datetimes(self, datetimes: np.ndarray, ts:Timescale):
        positions_at_datetimes = self.propogate_to_datetimes(datetimes, ts)
        subpoints_at_datetimes = {
            wgs84.subpoint_of(positions)
            for positions in positions_at_datetimes
        }

        lons = np.concat([
            subpoint.longitude.degrees
            for subpoint in subpoints_at_datetimes
        ])
        lats = np.concat([
            subpoint.latitude.degrees
            for subpoint in subpoints_at_datetimes
        ])
        return lons, lats


    def get_lon_lat_at_datetimes_with_lerp(self, datetimes: np.ndarray, ts: Timescale):
        # perform lerp between epochs to obtain weightings of each datetime
        datetimeranges = []
        for epoch_before, epoch_after in zip(
            [None, *self.epochs[:-1]],
            [*self.epochs[1:], None]
        ):
            datetimeranges.append(
                DateTimeRange(start=epoch_before, end=epoch_after)
            )

        lerp_contributions_by_epoch = np.array([
            [
                np.minimum(
                    np.clip( (datetimes - dtr.start) / (epoch - dtr.start) , 0, 1 ),
                    np.clip( (datetimes - dtr.end) / (epoch - dtr.end) , 0, 1)
                )
            ]
            for epoch, dtr in zip(self.epochs, datetimeranges)
        ])

        # get all positions from all epochs for all datetimes
        # SLOW, but easy to implement
        positions_by_epoch = [
            wgs84.subpoint_of(es.at(ts.from_datetimes(datetimes)))
            for es in self.tles
        ]
        lons_lats_by_epoch = [
            np.stack( pos.longitude.degrees, pos.latitude.degrees, -1 )
            for pos in positions_by_epoch
        ]
        positions = _get_mean_vector_on_sphere(lons_lats_by_epoch, axis=0, weights=lerp_contributions_by_epoch)
        return positions


    @classmethod
    def from_json_file(cls, fpath: str, ts: Timescale) -> Self:
        assert os.path.isfile(fpath), f"{fpath=} is not a valid file"
        with open(fpath, "r") as f:
            omm_json = json.load(f)

        if isinstance(omm_json, dict):
            tles = [EarthSatellite.from_omm(ts=ts, element_dict=omm_json)]
        elif isinstance(omm_json, list):
            tles = [
                EarthSatellite.from_omm(ts=ts, element_dict=omm)
                for omm in omm_json
            ]
        else:
            raise ValueError(omm_json)

        return cls(tles)
        

def _get_mean_vector_on_sphere(lons_lats_by_epoch, axis: int = 0, weights: np.ndarray = 1):
    lons_lats_rad = np.deg2rad(lons_lats_by_epoch)

    mx = np.sum( np.cos(lons_lats_rad[...,0])*np.cos(lons_lats[...,1])* weights, axis=axis ) / np.sum(weights, axis=axis)
    my = np.sum( np.sin(lons_lats_rad[...,0])*np.cos(lons_lats[...,1])* weights, axis=axis ) / np.sum(weights, axis=axis)
    mz = np.sum( np.sin(lons_lats_rad[...,1])*weights, axis=axis ) / np.sum(weights, axis=axis)
    norm = np.linalg.norm([mx,my,mx])
    mx /= norm
    my /= norm
    mz /= norm

    mlats_rad = np.arcsin(mz)
    mlons_rad = np.arctan2(my, mz)
    return np.rad2deg(mlons_rad), np.rad2deg(mlats_rad)



class DateTimeRange:
    def __init__(self, start: dt.datetime, end: dt.datetime, end_inclusive: bool = True):
        assert isinstance(self.start, dt.datetime, None)
        assert isinstance(self.end, dt.datetime, None)
        self.start = start
        self.end = end
        seld.end_inclusive = end_inclusive

    def __contains__(self, other: dt.datetime):
        cond_start = True
        if self.start is not None:
            cond_start = (other >= self.start)

        cond_end = True
        if self.end is not None:
            if self.end_inclusive:
                cond_end = (other <= self.end)
            else:
                cond_end = (other < self.end) 

        return cond_start & cond_end
