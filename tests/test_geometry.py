import numpy as np

from qr_assay.geometry import (
    data_module_mask,
    geometry_features,
    make_qr,
    transform_grid,
    transform_matrix,
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
