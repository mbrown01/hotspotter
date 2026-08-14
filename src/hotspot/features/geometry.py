"""Small, dependency-light geometry helpers shared by the feature detectors.

Kept separate so the chemistry code reads like chemistry, not linear algebra.
"""

from __future__ import annotations

import numpy as np


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 3D points (angstroms, given PDB coordinates)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by points a-b-c, in degrees.

    Used for H-bond D-H...A geometry when explicit hydrogens are present.
    """
    ba = np.asarray(a) - np.asarray(b)
    bc = np.asarray(c) - np.asarray(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def ring_centroid_and_normal(coords: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Centroid and unit normal of a planar ring, from its atom coordinates.

    The normal comes from the plane best-fit (SVD of the mean-centered points); its sign
    is arbitrary, which is fine because we only ever use the *angle between* two normals.
    """
    pts = np.asarray(coords, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # Smallest singular vector = direction of least spread = plane normal.
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    return centroid, normal


def ring_dihedral(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    """Angle between two ring planes (0-90 deg), via their normals.

    ~0 deg  -> parallel rings (face-to-face pi-stacking, "sandwich").
    ~90 deg -> perpendicular rings (edge-to-face, "T-shaped" aromatic contact).
    Both are real, favorable aromatic interactions; we report the angle so the user can
    tell them apart.
    """
    cosang = abs(float(np.dot(normal_a, normal_b)))
    return float(np.degrees(np.arccos(np.clip(cosang, 0.0, 1.0))))
