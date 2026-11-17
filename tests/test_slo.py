from genmedia_guard.slo import SLO, Sample, SlidingWindow, burn_rate, percentile


def test_percentile_nearest_rank():
    assert percentile([], 0.95) == 0.0
    assert percentile(list(range(1, 101)), 0.95) == 95
    assert percentile([5.0], 0.95) == 5.0


def test_window_stats_and_pruning():
    w = SlidingWindow(window_s=10)
    w.add(Sample(0, 100, True))
    w.add(Sample(5, 300, True, quality_ok=False))
    w.add(Sample(6, 50, False))
    s = w.stats(now=6)
    assert s.count == 3
    assert abs(s.error_rate - 1 / 3) < 1e-9
    assert s.quality_fail_rate == 0.5  # 1 of the 2 successful requests
    assert s.p95_ms == 300  # errors are excluded from latency
    # The first sample falls out of the window.
    assert w.stats(now=11).count == 2


def test_empty_window_is_all_zero():
    s = SlidingWindow(10).stats(now=100)
    assert (s.count, s.error_rate, s.p95_ms) == (0, 0.0, 0.0)


def test_burn_rate():
    slo = SLO(max_error_rate=0.01)
    assert burn_rate(0.01, slo) == 1.0
    assert burn_rate(0.05, slo) == 5.0
