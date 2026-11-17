import pytest

from genmedia_guard.metrics import Counter, Gauge, Histogram, Registry


def test_counter_renders_labels_and_rejects_decrease():
    c = Counter("req_total", "Requests.", ("version",))
    c.inc(version="v1")
    c.inc(2, version="v1")
    assert c.value(version="v1") == 3
    with pytest.raises(ValueError):
        c.inc(-1, version="v1")
    assert 'req_total{version="v1"} 3.0' in "\n".join(c.render())


def test_wrong_labels_are_rejected():
    c = Counter("x_total", "x", ("a",))
    with pytest.raises(ValueError):
        c.inc(b="1")


def test_label_values_are_escaped():
    g = Gauge("g", "g", ("name",))
    g.set(1, name='we"ird\nvalue')
    assert 'name="we\\"ird\\nvalue"' in "\n".join(g.render())


def test_histogram_is_cumulative_with_inf_sum_and_count():
    h = Histogram("lat_seconds", "Latency.", ("v",), buckets=(1.0, 2.0))
    for value in (0.5, 1.5, 5.0):
        h.observe(value, v="a")
    text = "\n".join(h.render())
    assert 'lat_seconds_bucket{v="a",le="1.0"} 1.0' in text
    assert 'lat_seconds_bucket{v="a",le="2.0"} 2.0' in text
    assert 'lat_seconds_bucket{v="a",le="+Inf"} 3.0' in text
    assert 'lat_seconds_sum{v="a"} 7.0' in text
    assert 'lat_seconds_count{v="a"} 3.0' in text


def test_registry_renders_help_and_type():
    r = Registry()
    r.register(Counter("a_total", "A.", ()))
    out = r.render()
    assert "# HELP a_total A." in out and "# TYPE a_total counter" in out
    assert out.endswith("\n")
