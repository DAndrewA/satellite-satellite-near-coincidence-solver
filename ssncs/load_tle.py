"""Module allowing the loading of TLE files or OMM files to create mutliple skyfield.sgp4lib.EarthSatellite instances"""

from skyfield.api import EarthSatellite
from skyfield.timelib import Timescale
from typing import Any
import os
import datetime as dt

type OMM_DICT = dict[str, Any]


def read_omm_json(fpath: str) -> OMM_DICT | list[OMM_DICT]:
    assert os.path.isfile(fpath), f"{fpath=} is not a valid file"
    with open(fpath, "r") as f:
        omm_json = json.load(f)
    return omm_json


def from_json(fpath: str, ts: Timescale):
    omm_json = read_omm_json(fpath)

    if isinstance(omm_json, dict):
        return EarthSatellite.from_omm(ts=ts, element_dict=omm_json)
    elif isinstance(omm_json, list):
        return from_OMM_dict(
            omm_dict = {
                dt.datetime.fromisoformat(omm["EPOCH"]): EarthSatellite.from_omm(ts=ts, element_dict=omm)
                for omm in omm_json
            }
        )
    else:
        raise ValueError(omm_json)
    
