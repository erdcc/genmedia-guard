import numpy as np

from genmedia_guard.fake_model import FakeDiffusionModel, ModelVersion
from genmedia_guard.quality import OutputValidator, psi, sharpness


def _img(version=None, seed=1):
    return FakeDiffusionModel(version or ModelVersion("t"))._render(seed)


def test_good_image_passes():
    report = OutputValidator().validate(_img())
    assert report.ok and report.reasons == ()


def test_blank_and_saturated_and_bad_shape_are_rejected():
    v = OutputValidator()
    assert v.validate(np.full((64, 64, 3), 127, np.uint8)).reasons == ("blank_output",)
    white = np.full((64, 64, 3), 255, np.uint8)
    assert "saturated_output" in v.validate(white).reasons
    assert v.validate(np.zeros((8, 8, 3), np.uint8)).reasons == ("bad_shape_or_dtype",)
    assert v.validate("not an image").reasons == ("bad_shape_or_dtype",)


def test_blur_lowers_sharpness():
    sharp = np.mean([sharpness(_img(seed=i)) for i in range(20)])
    blurry = np.mean([sharpness(_img(ModelVersion("b", blur=0.7), seed=i)) for i in range(20)])
    assert blurry < 0.85 * sharp


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(10, 2, 500)
    same = rng.normal(10, 2, 150)
    shifted = rng.normal(7, 2, 150)
    assert psi(base, same) < 0.1
    assert psi(base, shifted) > 0.25


def test_psi_handles_empty_and_constant_inputs():
    assert psi([], [1.0]) == 0.0
    assert psi([1.0] * 50, [1.0] * 20) == 0.0
    assert psi([1.0] * 50, [2.0] * 20) == float("inf")
