import numpy as np

from qr_assay.cluster import CORE_METRICS, ClusterAccumulator


def test_host_clustering_does_not_count_repeated_rows_as_independent():
    accumulator = ClusterAccumulator()
    positive = np.ones(len(CORE_METRICS), dtype=np.float64)
    negative = -np.ones(len(CORE_METRICS), dtype=np.float64)

    # 200 rows but only two independent host clusters. A row-wise standard error
    # would be tiny; the cluster-robust error must reflect two opposing hosts.
    for _ in range(100):
        accumulator.add(positive, host="host-a", source="crawl-a")
        accumulator.add(negative, host="host-b", source="crawl-b")

    summary = accumulator.summarize("host")
    assert all(row["n_matches"] == 200 for row in summary)
    assert all(row["cluster_count"] == 2 for row in summary)
    assert all(np.isclose(row["mean_difference"], 0.0) for row in summary)
    assert all(np.isclose(row["cr1_standard_error"], 1.0) for row in summary)
    assert all(not row["cluster_count_ge_20"] for row in summary)


def test_unique_hosts_reduce_to_nearly_ordinary_mean_uncertainty():
    accumulator = ClusterAccumulator()
    values = [-1.0, -0.5, 0.5, 1.0]
    for index, value in enumerate(values):
        vector = np.full(len(CORE_METRICS), value, dtype=np.float64)
        accumulator.add(vector, host=f"host-{index}", source=f"crawl-{index}")

    expected = np.std(values, ddof=1) / np.sqrt(len(values))
    summary = accumulator.summarize("host")
    assert all(row["cluster_count"] == len(values) for row in summary)
    assert all(np.isclose(row["cr1_standard_error"], expected) for row in summary)


def test_single_crawl_is_reported_as_non_estimable():
    accumulator = ClusterAccumulator()
    for index in range(10):
        accumulator.add(
            np.full(len(CORE_METRICS), float(index), dtype=np.float64),
            host=f"host-{index}",
            source="one-crawl",
        )
    summary = accumulator.summarize("source")
    assert all(row["cluster_count"] == 1 for row in summary)
    assert all(not row["estimable"] for row in summary)
    assert all(row["cr1_standard_error"] is None for row in summary)
