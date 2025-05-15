"""Module allowing the loading of TLE files or OMM files to create mutliple skyfield.sgp4lib.EarthSatellite instances"""

import skyfield.api import EarthSatellite
from typing import Any
import os
import datetime as dt

type OMM_DICT = dict[str, Any]

def from_OMM_dict(omm_dict: dict[Any, OMM_DICT]) -> dict[Any, EarthSatellite]:
    """Function to transform dictionary of OMM dictionaries into a dictionary of EarthSatellite instances with the same associated keys"""
    return {
        key: EarthSatellite.from_omm(omm)
        for key, omm in omm_dict.items()
    }


def from_OMM_JSON(fpath: str):
    assert os.path.isfile(fpath), f"{fpath=} is not a valid file"
    with open(fpath, "r") as f:
        omm_json = json.loads(f.read())

    if isinstance(omm_json, dict):
        return EarthSatellite.from_omm(omm_json)
    elif isinstance(omm_json, list):
        return from_OMM_dict(
            omm_dict = {
                dt.datetime.fromisoformat(v["EPOCH"]): v
                for v in omm_json
            }
        )
    else:
        raise ValueError(omm_json)
    
