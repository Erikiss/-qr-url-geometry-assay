import numpy as np

from qr_assay.geometry import geometry_features, make_qr, transform_matrix


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
    matrix, _ = make_qr("http://" + "a" * 56 + ".onion/", mask=1)
    inverse = transform_matrix(matrix, inverted=True)
    restored = transform_matrix(inverse, inverted=True)
    assert np.array_equal(restored, matrix)
    assert geometry_features(inverse)["density"] == 1.0 - geometry_features(matrix)["density"]


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
