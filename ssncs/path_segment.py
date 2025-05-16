"""Script containing class definitions for PathSegment class which defines motions in a 2d coordinate system, and contains methods for determining intersection"""

from __future__ import annotations
from functools import cached_property
from dataclasses import dataclass
from typing import Self
import numpy as np

@dataclass
class PathSegment:
    def __init__(self, start, end):
        assert start.shape == (2,), f"{start.shape=} should be (2,)"
        assert end.shape == (2,), f"{end.shape=} should be (2,)"
        self.start = start
        self.end = end

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
        return np.linalg.norm(np.array([
            self.dx[1],
            -self.dx[0]
        ]))

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
