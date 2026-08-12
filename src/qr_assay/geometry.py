from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import qrcode
from qrcode.constants import (
    ERROR_CORRECT_H,
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
)

ERROR_CORRECTION = {
    "L": ERROR_CORRECT_L,
    "M": ERROR_CORRECT_M,
    "Q": ERROR_CORRECT_Q,
    "H": ERROR_CORRECT_H,
}

D4_OPS = (
    ("r0", 0, "none"),
    ("r90", 90, "none"),
    ("r180", 180, "none"),
    ("r270", 270, "none"),
    ("mh", 0, "horizontal"),
    ("mv", 0, "vertical"),
    ("md", 0, "diagonal"),
    ("ma", 0, "anti_diagonal"),
)


def make_qr(
    payload: str, *, error_correction: str = "M", mask: int = 0, border: int = 0
) -> tuple[np.ndarray, int]:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECTION[error_correction.upper()],
        box_size=1,
        border=int(border),
        mask_pattern=int(mask),
    )
    qr.add_data(payload.encode("utf-8"), optimize=0)
    qr.make(fit=True)
    matrix = np.asarray(qr.get_matrix(), dtype=np.uint8)
    return matrix, int(qr.version)


def transform_matrix(
    matrix: np.ndarray,
    *,
    rotation: int = 0,
    reflection: str = "none",
    scale: int = 1,
    inverted: bool = False,
    group_element: str | None = None,
) -> np.ndarray:
    del group_element  # provenance label only; operation is encoded below
    result = np.rot90(matrix, k=(int(rotation) // 90) % 4)
    if reflection == "horizontal":
        result = np.fliplr(result)
    elif reflection == "vertical":
        result = np.flipud(result)
    elif reflection == "diagonal":
        result = result.T
    elif reflection == "anti_diagonal":
        result = np.fliplr(np.flipud(result)).T
    elif reflection != "none":
        raise ValueError(f"Unknown reflection {reflection!r}")
    if int(scale) > 1:
        result = np.repeat(np.repeat(result, int(scale), axis=0), int(scale), axis=1)
    if inverted:
        result = 1 - result
    return np.ascontiguousarray(result, dtype=np.uint8)


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def geometry_features(matrix: np.ndarray) -> dict[str, float | int | str]:
    data = np.asarray(matrix, dtype=np.uint8)
    height, width = data.shape
    density = float(data.mean())
    total = int(data.sum())
    yy, xx = np.indices(data.shape, dtype=np.float64)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    norm = max(width - 1, height - 1, 1)
    if total:
        centroid_x = float((xx * data).sum() / total)
        centroid_y = float((yy * data).sum() / total)
        dx = (xx - centroid_x) / norm
        dy = (yy - centroid_y) / norm
        cxx = float((data * dx * dx).sum() / total)
        cyy = float((data * dy * dy).sum() / total)
        cxy = float((data * dx * dy).sum() / total)
        covariance = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(covariance)
        radial = np.sqrt(((xx - center_x) / norm) ** 2 + ((yy - center_y) / norm) ** 2)
        radial_mean = float((data * radial).sum() / total)
        radial_std = float(np.sqrt((data * (radial - radial_mean) ** 2).sum() / total))
        angle = 0.5 * math.degrees(math.atan2(2.0 * cxy, cxx - cyy))
        anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / (eigenvalues.sum() + 1e-12))
    else:
        centroid_x = center_x
        centroid_y = center_y
        cxx = cyy = cxy = radial_mean = radial_std = angle = anisotropy = 0.0
        eigenvalues = np.array([0.0, 0.0])
    transition_h = _safe_mean(data[:, 1:] != data[:, :-1])
    transition_v = _safe_mean(data[1:, :] != data[:-1, :])
    packed = np.packbits(data, axis=None).tobytes()
    return {
        "height": height,
        "width": width,
        "black_modules": total,
        "density": density,
        "centroid_x": (centroid_x - center_x) / norm,
        "centroid_y": (centroid_y - center_y) / norm,
        "centroid_radius": math.hypot(centroid_x - center_x, centroid_y - center_y) / norm,
        "radial_mean": radial_mean,
        "radial_std": radial_std,
        "cov_xx": cxx,
        "cov_yy": cyy,
        "cov_xy": cxy,
        "cov_trace": cxx + cyy,
        "principal_angle_deg": angle,
        "anisotropy": anisotropy,
        # Axial orientation is modulo 180 degrees, so 2*theta is the correct
        # circular representation for inference.
        "orientation_cos2": anisotropy * math.cos(math.radians(2.0 * angle)),
        "orientation_sin2": anisotropy * math.sin(math.radians(2.0 * angle)),
        "transition_h": transition_h,
        "transition_v": transition_v,
        "symmetry_rot180": float((data == np.rot90(data, 2)).mean()),
        "symmetry_horizontal": float((data == np.fliplr(data)).mean()),
        "symmetry_vertical": float((data == np.flipud(data)).mean()),
        "matrix_sha256": hashlib.sha256(packed).hexdigest(),
    }


def transform_grid(config: dict[str, Any]):
    inversions = [bool(x) for x in config["transforms"]["inversions"]]
    scales = [int(x) for x in config["transforms"]["scales"]]
    if str(config["transforms"].get("group", "factorial")) == "d4":
        for group_element, rotation, reflection in D4_OPS:
            for scale in scales:
                for inverted in inversions:
                    yield {
                        "group_element": group_element,
                        "rotation": rotation,
                        "reflection": reflection,
                        "scale": scale,
                        "inverted": inverted,
                    }
        return
    for rotation in config["transforms"]["rotations"]:
        for reflection in config["transforms"]["reflections"]:
            for scale in scales:
                for inverted in inversions:
                    yield {
                        "group_element": None,
                        "rotation": int(rotation),
                        "reflection": str(reflection),
                        "scale": scale,
                        "inverted": inverted,
                    }
