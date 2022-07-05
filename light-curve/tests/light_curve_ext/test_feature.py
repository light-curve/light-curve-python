import inspect

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import light_curve.light_curve_ext as lc


def _feature_classes(module, exclude_parametric=True):
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.ismodule(obj):
            yield from _feature_classes(obj)
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, lc._FeatureEvaluator):
            continue
        # Skip classes with non-trivial constructors
        if exclude_parametric:
            try:
                obj()
            except TypeError:
                continue
        yield obj


non_param_feature_classes = frozenset(_feature_classes(lc, True))
all_feature_classes = frozenset(_feature_classes(lc, False))


def gen_lc(n, rng=None):
    rng = np.random.default_rng(rng)

    t = np.sort(rng.normal(0, 1, n))
    m = t.copy()
    sigma = np.full_like(t, 0.1)

    return t, m, sigma


@pytest.mark.parametrize("cls", non_param_feature_classes)
def test_negative_strides(cls):
    t = np.linspace(1, 0, 20)[::-2]
    m = np.exp(t)[:]
    err = np.random.uniform(0.1, 0.2, t.shape)
    obj = cls()
    obj(t, m, err)


@pytest.mark.parametrize("cls", non_param_feature_classes)
def test_float32_vs_float64(cls):
    rng = np.random.default_rng(0)
    n = 128

    t, m, sigma = gen_lc(n, rng=rng)
    obj = cls()

    results = [
        obj(t.astype(dtype), m.astype(dtype), sigma.astype(dtype), sorted=True) for dtype in [np.float32, np.float64]
    ]
    assert_allclose(*results, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cls", non_param_feature_classes)
def test_many_vs_call(cls):
    rng = np.random.default_rng(0)
    n_obs = 128
    n_lc = 128

    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]
    obj = cls()

    call = np.stack([obj(*lc, sorted=True) for lc in lcs])
    many = obj.many(lcs, sorted=True, n_jobs=2)
    assert_array_equal(many, call)


def test_fill_value_not_enough_observations():
    n = 1
    t = np.linspace(0.0, 1.0, n)
    m = t.copy()
    fill_value = -100.0
    sigma = np.ones_like(t)
    feature = lc.Kurtosis()
    with pytest.raises(ValueError):
        feature(t, m, sigma, fill_value=None)
    assert_array_equal(feature(t, m, sigma, fill_value=fill_value), fill_value)


@pytest.mark.parametrize("cls", all_feature_classes)
def test_nonempty_docstring(cls):
    assert len(cls.__doc__) > 10


@pytest.mark.parametrize("cls", non_param_feature_classes)
def test_check_t(cls):
    n_obs = 128
    t, m, sigma = gen_lc(n_obs)
    t[0] = np.nan
    feature = cls()
    with pytest.raises(ValueError):
        feature(t, m, sigma, check=True)
    t[0] = np.inf
    with pytest.raises(ValueError):
        feature(t, m, sigma, check=True)


@pytest.mark.parametrize("cls", non_param_feature_classes)
def test_check_m(cls):
    n_obs = 128
    t, m, sigma = gen_lc(n_obs)
    m[0] = np.nan
    feature = cls()
    with pytest.raises(ValueError):
        feature(t, m, sigma, check=True)
    m[0] = np.inf
    with pytest.raises(ValueError):
        feature(t, m, sigma, check=True)


# We need evaluators which use sigma
@pytest.mark.parametrize("cls", (lc.ExcessVariance, lc.LinearFit, lc.ReducedChi2, lc.StetsonK, lc.WeightedMean))
def test_check_sigma(cls):
    n_obs = 128
    t, m, sigma = gen_lc(n_obs)
    sigma[0] = np.nan
    feature = cls()
    with pytest.raises(ValueError):
        feature(t, m, sigma, check=True)
    # infinite values are allowed for sigma
    sigma[0] = np.inf
    feature(t, m, sigma, check=True)
