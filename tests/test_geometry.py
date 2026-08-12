import numpy as np
import qrcode

from qr_assay.geometry import (
    ERROR_CORRECTION,
    codeword_region_masks,
    data_module_mask,
    geometry_features,
    make_qr,
    transform_grid,
    transform_matrix,
    unmask_data_modules,
)


def test_rotation_and_reflection_preserve_density():
    matrix, _ = make_qr("https://example.invalid/", mask=3)
    base = geometry_features(matrix)
    for rotation in (0, 90, 180, 270):
        for reflection in (
            "none",
            "horizontal",
            "vertical",
            "diagonal",
            "anti_diagonal",
        ):
            transformed = transform_matrix(matrix, rotation=rotation, reflection=reflection)
            assert geometry_features(transformed)["density"] == base["density"]


def test_inversion_is_an_involution_and_complements_density():
    matrix, version = make_qr("http://" + "a" * 56 + ".onion/", mask=1)
    region = data_module_mask(version)
    inverse = transform_matrix(matrix, inverted=True)
    restored = transform_matrix(inverse, inverted=True)
    assert np.array_equal(restored, matrix)
    assert np.isclose(
        geometry_features(inverse)["density"],
        1.0 - geometry_features(matrix)["density"],
        atol=1e-15,
        rtol=0.0,
    )
    assert np.isclose(
        geometry_features(inverse, region)["density"],
        1.0 - geometry_features(matrix, region)["density"],
        atol=1e-15,
        rtol=0.0,
    )


def test_data_module_mask_excludes_fixed_qr_function_patterns():
    matrix, version = make_qr("https://example.invalid/data-region", mask=4)
    region = data_module_mask(version)
    assert region.shape == matrix.shape
    assert 0 < int(region.sum()) < region.size
    # Finder pattern corners are reserved, while there must be data/ECC positions.
    assert region[0, 0] == 0
    assert region[-1, 0] == 0
    assert region[0, -1] == 0
    assert np.any(region == 1)


def test_codeword_regions_are_disjoint_and_exactly_partition_free_modules():
    for ecc in ("L", "M", "Q", "H"):
        matrix, version = make_qr(
            "https://example.invalid/codeword-partition", mask=3, error_correction=ecc
        )
        free = data_module_mask(version).astype(bool)
        data_region, ecc_region, remainder_region = codeword_region_masks(version, ecc)
        data_region = data_region.astype(bool)
        ecc_region = ecc_region.astype(bool)
        remainder_region = remainder_region.astype(bool)

        assert data_region.shape == matrix.shape
        assert not np.any(data_region & ecc_region)
        assert not np.any(data_region & remainder_region)
        assert not np.any(ecc_region & remainder_region)
        assert np.array_equal(data_region | ecc_region | remainder_region, free)
        # ISO QR versions have only a small tail of remainder bits after codewords.
        assert int(remainder_region.sum()) < 8
        assert int(data_region.sum()) > 0
        assert int(ecc_region.sum()) > 0


def test_unmasking_recovers_same_free_module_bitstream_for_all_eight_masks():
    payload = "https://example.invalid/the/same/payload?q=qr"
    recovered = []
    version_seen = None
    free = None
    for mask in range(8):
        matrix, version = make_qr(payload, mask=mask, error_correction="M")
        if version_seen is None:
            version_seen = version
            free = data_module_mask(version).astype(bool)
        assert version == version_seen
        unmasked = unmask_data_modules(matrix, version, mask)
        recovered.append(unmasked[free])
    assert all(np.array_equal(recovered[0], values) for values in recovered[1:])


def test_fixed_function_region_is_payload_independent_at_fixed_version_ecc_and_mask():
    matrix_a, version_a = make_qr("https://example.invalid/aaaaaaaaaaaaaaaa", mask=5)
    matrix_b, version_b = make_qr("https://example.invalid/bbbbbbbbbbbbbbbb", mask=5)
    assert version_a == version_b
    free = data_module_mask(version_a).astype(bool)
    assert np.array_equal(matrix_a[~free], matrix_b[~free])


def test_make_qr_forces_byte_mode_instead_of_encoder_auto_mode():
    payload = "1" * 40
    _, forced_version = make_qr(payload, mask=0, error_correction="M")

    auto = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECTION["M"],
        box_size=1,
        border=0,
        mask_pattern=0,
    )
    auto.add_data(payload.encode("utf-8"), optimize=0)
    auto.make(fit=True)

    # A numeric-only payload is much denser in QR numeric mode. If this test
    # fails, the assay has silently stopped forcing the intended byte-mode arm.
    assert forced_version > int(auto.version)


def test_scale_is_nearest_neighbor():
    matrix, _ = make_qr("https://example.invalid/", mask=0)
    scaled = transform_matrix(matrix, scale=2)
    assert scaled.shape == (matrix.shape[0] * 2, matrix.shape[1] * 2)
    assert np.array_equal(scaled[::2, ::2], matrix)


def test_mask_and_version_are_explicit():
    matrix_a, version_a = make_qr("https://example.invalid/a", mask=0)
    matrix_b, version_b = make_qr("https://example.invalid/a", mask=7)
    assert version_a == version_b
    assert matrix_a.shape == matrix_b.shape
    assert not np.array_equal(matrix_a, matrix_b)


def test_d4_grid_has_exactly_eight_unique_square_symmetries():
    config = {
        "transforms": {
            "group": "d4",
            "rotations": [0],
            "reflections": ["none"],
            "scales": [1],
            "inversions": [False],
        }
    }
    ops = list(transform_grid(config))
    assert len(ops) == 8
    assert len({op["group_element"] for op in ops}) == 8
    matrix = np.array([[0, 1, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)
    rendered = {transform_matrix(matrix, **op).tobytes() for op in ops}
    assert len(rendered) == 8


def test_orientation_embedding_is_axial_not_linear_angle():
    matrix, _ = make_qr("https://example.invalid/axial", mask=2)
    f0 = geometry_features(matrix)
    f180 = geometry_features(np.rot90(matrix, 2))
    assert np.isclose(f0["orientation_cos2"], f180["orientation_cos2"])
    assert np.isclose(f0["orientation_sin2"], f180["orientation_sin2"])
