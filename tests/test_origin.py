from qr_assay.sampling import _canonicalize_payload


def test_origin_strips_scheme_port_path_and_query():
    value = "http://example.org:8080/a/b?q=1"
    assert _canonicalize_payload(value, granularity="origin", scheme_policy="strip") == "example.org/"
    assert (
        _canonicalize_payload(value, granularity="origin", scheme_policy="https")
        == "https://example.org/"
    )


def test_full_url_arm_preserves_port():
    value = "http://example.org:8080/a/b?q=1"
    assert (
        _canonicalize_payload(value, granularity="url", scheme_policy="strip")
        == "example.org:8080/a/b?q=1"
    )
