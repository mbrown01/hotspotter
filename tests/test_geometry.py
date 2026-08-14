"""Pure geometry unit tests — no structures, no network."""

from __future__ import annotations

import numpy as np

from hotspotter.features.geometry import (
    angle,
    distance,
    ring_centroid_and_normal,
    ring_dihedral,
)


def test_distance():
    assert distance((0, 0, 0), (3, 4, 0)) == 5.0


def test_angle_right():
    # a-b-c with a right angle at b
    assert abs(angle((1, 0, 0), (0, 0, 0), (0, 1, 0)) - 90.0) < 1e-6


def test_angle_straight():
    # Tolerance is loose at the antiparallel edge: the zero-length-safety epsilon in the
    # normalization nudges cos(theta) a hair past -1. 0.01 deg is far below any physical
    # relevance (H-bond angle cutoffs are in whole degrees).
    assert abs(angle((1, 0, 0), (0, 0, 0), (-1, 0, 0)) - 180.0) < 0.01


def test_ring_normal_is_perpendicular_to_plane():
    # A square in the z=0 plane -> normal should be +/- z.
    coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    centroid, normal = ring_centroid_and_normal(coords)
    assert np.allclose(centroid, (0.5, 0.5, 0.0))
    assert abs(abs(normal[2]) - 1.0) < 1e-6  # points along z


def test_ring_dihedral_parallel_and_perpendicular():
    z = np.array([0, 0, 1.0])
    x = np.array([1.0, 0, 0])
    assert ring_dihedral(z, z) < 1e-6           # parallel -> 0 deg
    assert abs(ring_dihedral(z, x) - 90.0) < 1e-6  # perpendicular -> 90 deg
