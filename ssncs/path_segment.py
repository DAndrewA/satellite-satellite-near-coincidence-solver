"""Script containing class definitions for PathSegment class which defines motions in a 2d coordinate system, and contains methods for determining intersection"""

from __future__ import annotations
from functools import cached_property
from dataclasses import dataclass, asdict
import datetime as dt
from typing import Self
import numpy as np
import cartopy.crs as ccrs

class PathSegment:
    def __init__(self, start, end, crs: ccrs.CRS):
        assert start.shape == (2,), f"{start.shape=} should be (2,)"
        assert end.shape == (2,), f"{end.shape=} should be (2,)"
        assert isinstance(crs, ccrs.CRS), f"{type(crs)=} should be a subclass of CRS"
        self.start = start
        self.end = end
        self.crs = crs

    @classmethod
    def from_xy_arrays(cls, x, y) -> list[Self]:
        assert x.ndim == 1, f"{x.ndim=} must be 1"
        assert y.ndim == 1, f"{y.ndim=} must be 1"
        assert x.shape == y.shape, f"{x.shape=} must be equal to {y.shape=}"
        return (
            cls(
                start=np.array([x[i], y[i]]),
                end=np.array([x[i+1], y[i+1]])
            )
            for i in range(len(x)-1)
        )

    @cached_property
    def dx(self) -> np.ndarray:
        return self.end - self.start

    @cached_property
    def n(self) -> np.ndarray:
        """Right-pointing normal with respect to dx"""
        n = np.array([
            self.dx[1],
            -self.dx[0]
        ])
        return n / np.linalg.norm(n)

    def intersects_plane_defined_by(self, other: PathSegment) -> bool:
        """Function to assess if the ray defined by self intersects the plane defined by other.
        This is achieved by determining the sign difference between displacement of the start and end vector and the plane defined by the other path segment.
        """
        assert isinstance(other, PathSegment), f"{type(other)=} must be instance of PathSegment"
        start_plane_displacement = np.dot( (self.start - other.start), other.n )
        end_plane_displacement = np.dot( (self.end - other.start), other.n )
        return (start_plane_displacement * end_plane_displacement) <= 0

    def intersects(self, other: PathSegment) -> bool:
        """Function to determine if two PathSegment objects intersect in 2d space"""
        assert isinstance(other, PathSegment), f"{type(other)=} must be instance of PathSegment"
        return self.intersects_plane_defined_by(other) and other.intersects_plane_defined_by(self)

    def transform_crs(self, new_crs:ccrs.CRS) -> Self:
        new_start = np.array(new_crs.transform_point(self.start[0], self.start[1], self.crs))
        new_end = np.array(new_crs.transform_point(self.end[0], self.end[1], self.crs))
        return type(self)(start=new_start, end=new_end, crs=new_crs)

    def to_dict(self) -> dict:
        return dict(
            start = self.start,
            end = self.end
        )


@dataclass(frozen=True, kw_only=True)
class Intersection:
    def __init__(self, ps1, ps2, dt1: dt.datetime | None, dt2: dt.datetime | None):
        assert isinstance(ps1, PathSegment), f"{type(ps1)=} should be an instance of PathSegment"
        assert isinstance(ps2, PathSegment), f"{type(ps2)=} should be an instance of PathSegment"
        self.ps1 = ps1
        self.ps2 = ps2
        selt.dt1 = dt1
        self.dt2 = dt2
