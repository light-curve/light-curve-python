import numpy as np

from light_curve.light_curve_ext import Eta, Mean
from light_curve.light_curve_ext import _FeatureEvaluator as _RustBaseFeature
from light_curve.light_curve_py import Amplitude, Extractor, OtsuSplit
from light_curve.light_curve_py.features._base import BaseFeature as _PythonBaseFeature


def test_extractor_all_python():
    t = np.linspace(0.0, 1.0, 50)
    perfect_m = 1e3 * t + 1e2
    err = np.sqrt(perfect_m)
    m = perfect_m + np.random.normal(0, err)

    otsusplit = OtsuSplit()
    amplitude = Amplitude()

    assert isinstance(otsusplit, _PythonBaseFeature)
    assert isinstance(amplitude, _PythonBaseFeature)

    extractor = Extractor(otsusplit, amplitude)
    actual = extractor(t, m, err)
    desired = np.concatenate([np.atleast_1d(otsusplit(t, m, err)), np.atleast_1d(amplitude(t, m, err))])
    assert np.all(actual == desired)


def test_extractor_all_rust():
    t = np.linspace(0.0, 1.0, 50)
    perfect_m = 1e3 * t + 1e2
    err = np.sqrt(perfect_m)
    m = perfect_m + np.random.normal(0, err)
    mean = Mean()
    eta = Eta()

    assert isinstance(mean, _RustBaseFeature)
    assert isinstance(eta, _RustBaseFeature)

    extractor = Extractor(mean, eta)
    actual = extractor(t, m, err)
    desired = np.concatenate([np.atleast_1d(mean(t, m, err)), np.atleast_1d(eta(t, m, err))])
    assert np.all(actual == desired)


def test_extractor_mixed_python_rust():
    t = np.linspace(0.0, 1.0, 50)
    perfect_m = 1e3 * t + 1e2
    err = np.sqrt(perfect_m)
    m = perfect_m + np.random.normal(0, err)
    eta = Eta()
    otsusplit = OtsuSplit()

    assert isinstance(eta, _RustBaseFeature)
    assert isinstance(otsusplit, _PythonBaseFeature)

    extractor = Extractor(eta, otsusplit)
    actual = extractor(t, m, err)
    desired = np.concatenate([np.atleast_1d(eta(t, m, err)), np.atleast_1d(otsusplit(t, m, err))])
    assert np.all(actual == desired)
