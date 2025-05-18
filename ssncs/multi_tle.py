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
        
