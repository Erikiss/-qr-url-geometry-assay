from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Any

import numpy as np
import qrcode
from qrcode import base, util
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
    """Encode one payload in explicitly forced QR byte mode.

    Forcing byte mode removes a hidden encoder-choice confound: the same lexical
    comparison must not silently switch between numeric/alphanumeric/byte modes.
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECTION[error_correction.upper()],
        box_size=1,
        border=int(border),
        mask_pattern=int(mask),
    )
    qr.add_data(
        util.QRData(
            payload.encode("utf-8"),
            mode=util.MODE_8BIT_BYTE,
            check_data=False,
        )
    )
    qr.make(fit=True)
    matrix = np.asarray(qr.get_matrix(), dtype=np.uint8)
    return matrix, int(qr.version)


@lru_cache(maxsize=40)
def data_module_mask(version: int) -> np.ndarray:
    """Return 1 exactly where QR data/ECC/remainder modules may be mapped.

    The qrcode library builds function patterns before `map_data`. We replay that
    pre-map construction with test type information, then mark the remaining
    `None` cells. This avoids hand-maintaining finder/timing/alignment/version
    coordinates and keeps the region definition tied to the pinned encoder.
    """
    qr = qrcode.QRCode(
        version=int(version),
        error_correction=ERROR_CORRECT_M,
        box_size=1,
        border=0,
        mask_pattern=0,
    )
    qr.modules_count = int(version) * 4 + 17
    qr.modules = [[None] * qr.modules_count for _ in range(qr.modules_count)]
    qr.setup_position_probe_pattern(0, 0)
    qr.setup_position_probe_pattern(qr.modules_count - 7, 0)
    qr.setup_position_probe_pattern(0, qr.modules_count - 7)
    qr.setup_position_adjust_pattern()
    qr.setup_timing_pattern()
    qr.setup_type_info(True, 0)
    if int(version) >= 7:
        qr.setup_type_number(True)
    result = np.asarray([[cell is None for cell in row] for row in qr.modules], dtype=np.uint8)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=160)
def codeword_region_masks(
    version: int, error_correction: str = "M"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split mapped QR modules into data-codeword, RS-ECC and remainder regions.

    `python-qrcode==8.2` emits all interleaved data codewords first and all
    Reed-Solomon error-correction codewords second. Its `map_data` routine then
    consumes those bits in a deterministic zig-zag over the free modules. We
    replay only that traversal, assigning each free module by bit index.

    The *data-codeword* region is intentionally not called a pure payload region:
    it includes mode/count framing, payload bytes, terminator/alignment padding,
    and alternating QR pad codewords. The separation nevertheless removes the
    downstream Reed-Solomon parity region as a distinct source of geometry.
    """
    ecc_name = error_correction.upper()
    ecc = ERROR_CORRECTION[ecc_name]
    rs_blocks = base.rs_blocks(int(version), ecc)
    data_codeword_bits = sum(block.data_count for block in rs_blocks) * 8
    total_codeword_bits = sum(block.total_count for block in rs_blocks) * 8

    free = data_module_mask(int(version)).astype(bool)
    data_region = np.zeros_like(free, dtype=np.uint8)
    ecc_region = np.zeros_like(free, dtype=np.uint8)
    remainder_region = np.zeros_like(free, dtype=np.uint8)

    modules_count = free.shape[0]
    inc = -1
    row = modules_count - 1
    bit_index = 0

    for col in range(modules_count - 1, 0, -2):
        if col <= 6:
            col -= 1
        col_range = (col, col - 1)
        while True:
            for current_col in col_range:
                if free[row, current_col]:
                    if bit_index < data_codeword_bits:
                        data_region[row, current_col] = 1
                    elif bit_index < total_codeword_bits:
                        ecc_region[row, current_col] = 1
                    else:
                        remainder_region[row, current_col] = 1
                    bit_index += 1
            row += inc
            if row < 0 or modules_count <= row:
                row -= inc
                inc = -inc
                break

    if bit_index != int(free.sum()):
        raise AssertionError("QR codeword traversal did not consume all free modules")
    if int(data_region.sum()) != data_codeword_bits:
        raise AssertionError("QR data-codeword region has an unexpected size")
    if int(ecc_region.sum()) != total_codeword_bits - data_codeword_bits:
        raise AssertionError("QR ECC region has an unexpected size")
    if not np.array_equal(
        (data_region | ecc_region | remainder_region).astype(np.uint8), free.astype(np.uint8)
    ):
        raise AssertionError("QR codeword regions do not partition the free-module region")

    for region in (data_region, ecc_region, remainder_region):
        region.setflags(write=False)
    return data_region, ecc_region, remainder_region


@lru_cache(maxsize=320)
def _qr_mask_toggle_matrix(version: int, mask: int) -> np.ndarray:
    region = data_module_mask(int(version)).astype(bool)
    mask_func = util.mask_func(int(mask))
    toggle = np.zeros(region.shape, dtype=np.uint8)
    for row, col in np.argwhere(region):
        if mask_func(int(row), int(col)):
            toggle[row, col] = 1
    toggle.setflags(write=False)
    return toggle


def unmask_data_modules(matrix: np.ndarray, version: int, mask: int) -> np.ndarray:
    """Undo only the QR mask on data/ECC/remainder modules.

    Fixed finder/timing/format/version patterns are left untouched. This is a
    diagnostic representation of the encoded bitstream geometry, not a rendered
    QR code that should be fed to a scanner or model.
    """
    data = np.asarray(matrix, dtype=np.uint8)
    toggle = _qr_mask_toggle_matrix(int(version), int(mask))
    if data.shape != toggle.shape:
        raise ValueError("unmask_data_modules requires a border-free QR matrix")
    return np.ascontiguousarray(data ^ toggle, dtype=np.uint8)


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


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def geometry_features(
    matrix: np.ndarray, region_mask: np.ndarray | None = None
) -> dict[str, float | int | str]:
    data = np.asarray(matrix, dtype=np.uint8)
    height, width = data.shape
    if region_mask is None:
        region = np.ones(data.shape, dtype=bool)
    else:
        region = np.asarray(region_mask, dtype=bool)
        if region.shape != data.shape:
            raise ValueError("region_mask must have the same shape as matrix")
    weights = data * region
    region_modules = int(region.sum())
    total = int(weights.sum())
    density = _safe_ratio(total, region_modules)
    yy, xx = np.indices(data.shape, dtype=np.float64)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    norm = max(width - 1, height - 1, 1)
    if total:
        centroid_x = float((xx * weights).sum() / total)
        centroid_y = float((yy * weights).sum() / total)
        dx = (xx - centroid_x) / norm
        dy = (yy - centroid_y) / norm
        cxx = float((weights * dx * dx).sum() / total)
        cyy = float((weights * dy * dy).sum() / total)
        cxy = float((weights * dx * dy).sum() / total)
        covariance = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)
        eigenvalues = np.linalg.eigvalsh(covariance)
        radial = np.sqrt(((xx - center_x) / norm) ** 2 + ((yy - center_y) / norm) ** 2)
        radial_mean = float((weights * radial).sum() / total)
        radial_std = float(np.sqrt((weights * (radial - radial_mean) ** 2).sum() / total))
        angle = 0.5 * math.degrees(math.atan2(2.0 * cxy, cxx - cyy))
        anisotropy = float((eigenvalues[-1] - eigenvalues[0]) / (eigenvalues.sum() + 1e-12))
    else:
        centroid_x = center_x
        centroid_y = center_y
        cxx = cyy = cxy = radial_mean = radial_std = angle = anisotropy = 0.0

    valid_h = region[:, 1:] & region[:, :-1]
    valid_v = region[1:, :] & region[:-1, :]
    transition_h = _safe_ratio(((data[:, 1:] != data[:, :-1]) & valid_h).sum(), valid_h.sum())
    transition_v = _safe_ratio(((data[1:, :] != data[:-1, :]) & valid_v).sum(), valid_v.sum())

    def symmetry_score(other_data: np.ndarray, other_region: np.ndarray) -> float:
        valid = region & other_region
        return _safe_ratio(((data == other_data) & valid).sum(), valid.sum())

    packed = np.packbits(data, axis=None).tobytes()
    return {
        "height": height,
        "width": width,
        "region_modules": region_modules,
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
        "orientation_cos2": anisotropy * math.cos(math.radians(2.0 * angle)),
        "orientation_sin2": anisotropy * math.sin(math.radians(2.0 * angle)),
        "transition_h": transition_h,
        "transition_v": transition_v,
        "symmetry_rot180": symmetry_score(np.rot90(data, 2), np.rot90(region, 2)),
        "symmetry_horizontal": symmetry_score(np.fliplr(data), np.fliplr(region)),
        "symmetry_vertical": symmetry_score(np.flipud(data), np.flipud(region)),
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
