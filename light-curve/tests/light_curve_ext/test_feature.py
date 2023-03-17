import copy
import inspect
import pickle

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


def construct_example_objects(cls, *, parametric_variants=1, rng=None):
    # Extractor is special
    if cls is lc.Extractor:
        return [cls(lc.BeyondNStd(1.5), lc.LinearFit())]

    # No mandatory arguments
    if not hasattr(cls, "__getnewargs__"):
        return [cls()]

    args = cls.__getnewargs__()  # default mandatory arguments
    # Add Mean feature for metafeatures
    args = [[lc.Mean()] if arg == () else arg for arg in args]

    objects = [cls(*args)]
    # Nothing to mutate
    if not any(isinstance(arg, float) for arg in args):
        return objects

    # Mutate floats
    rng = np.random.default_rng(rng)
    for _ in range(1, parametric_variants):
        mutated_args = [
            arg * rng.uniform(0.9, 1.1) + rng.uniform(0.0, 1e-3) if isinstance(arg, float) else arg for arg in args
        ]
        objects.append(cls(*mutated_args))
    return objects


def gen_feature_evaluators(*, parametric_variants=0, rng=None):
    if parametric_variants == 0:
        for cls in non_param_feature_classes:
            yield cls()
        return
    rng = np.random.default_rng(rng)
    for cls in all_feature_classes:
        yield from construct_example_objects(cls, parametric_variants=parametric_variants, rng=rng)


def gen_lc(n, rng=None):
    rng = np.random.default_rng(rng)

    t = np.sort(rng.normal(0, 1, n))
    m = t.copy()
    sigma = np.full_like(t, 0.1)

    return t, m, sigma


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=2))
def test_negative_strides(feature):
    t = np.linspace(1, 0, 20)[::-2]
    m = np.exp(t)[:]
    err = np.random.uniform(0.1, 0.2, t.shape)
    feature(t, m, err)


# We don't want *Fit features here: not precise
@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=0))
def test_float32_vs_float64(feature):
    rng = np.random.default_rng(0)
    n = 128

    t, m, sigma = gen_lc(n, rng=rng)

    results = [
        feature(t.astype(dtype), m.astype(dtype), sigma.astype(dtype), sorted=True)
        for dtype in [np.float32, np.float64]
    ]
    assert_allclose(*results, rtol=1e-5, atol=1e-5)


# We don't want *Fit features here: too slow
@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=0))
def test_many_vs_call(feature):
    rng = np.random.default_rng(0)
    n_obs = 128
    n_lc = 128

    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    call = np.stack([feature(*lc, sorted=True) for lc in lcs])
    many = feature.many(lcs, sorted=True, n_jobs=2)
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


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=2))
def test_check_t(feature):
    n_obs = 128
    t, m, sigma = gen_lc(n_obs)
    t[0] = np.nan
    with pytest.raises(ValueError):
        feature(t, m, sigma, check=True)
    t[0] = np.inf
    with pytest.raises(ValueError):
        feature(t, m, sigma, check=True)


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=2))
def test_check_m(feature):
    n_obs = 128
    t, m, sigma = gen_lc(n_obs)
    m[0] = np.nan
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


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=5, rng=None))
@pytest.mark.parametrize("pickle_protocol", tuple(range(2, pickle.HIGHEST_PROTOCOL + 1)))
def test_pickling(feature, pickle_protocol):
    n_obs = 128
    data = gen_lc(n_obs)
    values = feature(*data)

    b = pickle.dumps(feature, protocol=pickle_protocol)
    new_feature = pickle.loads(b)

    new_values = new_feature(*data)
    assert_array_equal(values, new_values)


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=5, rng=None))
def test_copy_deepcopy(feature):
    n_obs = 128
    data = gen_lc(n_obs)
    values = feature(*data)

    copied = copy.copy(feature)
    values_copied = copied(*data)
    assert_array_equal(values, values_copied)

    deepcopied = copy.deepcopy(feature)
    values_deepcopied = deepcopied(*data)
    assert_array_equal(values, values_deepcopied)


PICKLE_BENCHMARK_FEATURES = [
    lc.Amplitude(),  # no parameters
    lc.BeyondNStd(1.5),  # parametric
    lc.Extractor(  # large
        lc.Amplitude(),
        lc.BeyondNStd(2.0),
        lc.Bins(
            [lc.Kurtosis(), lc.LinearTrend(), lc.WeightedMean()],
            window=2.0,
            offset=59500.5,
        ),
        lc.Periodogram(features=[lc.InterPercentileRange(0.01)], peaks=5, max_freq_factor=12.0),
    ),
]


@pytest.mark.parametrize("feature", PICKLE_BENCHMARK_FEATURES)
def test_benchmark_pickle_loads(feature, benchmark):
    b = pickle.dumps(feature, protocol=pickle.HIGHEST_PROTOCOL)
    benchmark(pickle.loads, b)


@pytest.mark.parametrize("feature", PICKLE_BENCHMARK_FEATURES)
def test_benchmark_pickle_dumps(feature, benchmark):
    benchmark(pickle.dumps, feature, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("feature", PICKLE_BENCHMARK_FEATURES)
def test_benchmark_copy(feature, benchmark):
    benchmark(copy.copy, feature)


@pytest.mark.parametrize("feature", PICKLE_BENCHMARK_FEATURES)
def test_benchmark_deepcopy(feature, benchmark):
    benchmark(copy.deepcopy, feature)


# We do not check pure MCMC because it requires a lot of iterations and would be too slow
@pytest.mark.parametrize("algo", ("lmsder", "mcmc-lmsder"))
def test_bazin_fit_precise(algo):
    bazin = lc.BazinFit(algo)

    true_params = np.array([10.0, -2.0, 10.0, 10.0, 25.0])
    t = np.linspace(-50.0, 120.0, 1000)
    flux = bazin.model(t, true_params)
    fluxerr = np.ones_like(t)

    *params, reduced_chi2 = bazin(t, flux, fluxerr)
    assert_allclose(true_params, params, rtol=1e-4)  # tolerance set to underlying algorithms
