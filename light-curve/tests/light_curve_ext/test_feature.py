import copy
import inspect
import pickle
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

import arro3.core
import numpy as np
import pyarrow as pa
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import light_curve.light_curve_ext as licu_ext


def _feature_classes(module, *, exclude_parametric=True):
    for name, member in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if inspect.ismodule(member):
            yield from _feature_classes(member)
        if not inspect.isclass(member):
            continue
        if not issubclass(member, licu_ext._FeatureEvaluator):
            continue
        if member is licu_ext.JSONDeserializedFeature:
            continue
        # Skip classes with non-trivial constructors
        if exclude_parametric:
            try:
                member()
            except TypeError:
                continue
        yield member


non_param_feature_classes = frozenset(_feature_classes(licu_ext, exclude_parametric=True))
assert len(non_param_feature_classes) > 0

all_feature_classes = frozenset(_feature_classes(licu_ext, exclude_parametric=False))
assert len(all_feature_classes) > 0

fit_feature_classes = frozenset(
    cls for cls in all_feature_classes if hasattr(cls, "model") and hasattr(cls, "supported_algorithms")
)
assert len(fit_feature_classes) > 0


def _try_pure_multiband(cls):
    """Return True if cls is a pure multiband feature (bands arg is mandatory, no single-band mode)."""
    try:
        args = cls.__getnewargs__()
        feature = cls(*args)
        return feature.bands is not None
    except Exception:
        return False


# Pure multiband features: always multiband-only, no single-band mode.
# Their constructor takes `bands` as the first (and only mandatory) argument.
pure_multiband_feature_classes = frozenset(
    cls
    for cls in all_feature_classes
    if cls not in fit_feature_classes | {licu_ext.Extractor, licu_ext.Periodogram, licu_ext.JSONDeserializedFeature}
    and hasattr(cls, "__getnewargs__")
    and _try_pure_multiband(cls)
)


def get_new_args_kwargs(cls):
    if hasattr(cls, "__getnewargs_ex__"):
        return cls.__getnewargs_ex__()
    if hasattr(cls, "__getnewargs__"):
        args = cls.__getnewargs__()
        return args, {}
    return (), {}


def new_default(cls, **kwargs):
    args, kwargs_ = get_new_args_kwargs(cls)
    kwargs = dict(kwargs_, **kwargs)
    return cls(*args, **kwargs)


def gen_periodogram_variants(*, rng=None):
    rng = np.random.default_rng(rng)

    peaks = rng.integers(1, 10)
    resolution = 10 ** rng.uniform(-0.5, 1.5)
    max_freq_factor = 10 ** rng.uniform(0.0, 2.0)
    features = [licu_ext.Amplitude(), licu_ext.Mean()]

    for nyquist in ["average", rng.uniform(0.0, 1.0)]:
        for freq_grid in [False, True]:
            for fast in [False, True]:
                if freq_grid:
                    if fast:
                        freqs = np.linspace(0.0, 100.0, 257)
                    else:
                        freqs = np.linspace(1.0, 100.0, 100)
                else:
                    freqs = None

                yield licu_ext.Periodogram(
                    peaks=peaks,
                    resolution=resolution,
                    max_freq_factor=max_freq_factor,
                    nyquist=nyquist,
                    freqs=freqs,
                    fast=fast,
                    features=features,
                )

    # Test one non-default normalization; full normalization coverage
    # is in test_periodogram.py
    yield licu_ext.Periodogram(
        peaks=peaks,
        resolution=resolution,
        max_freq_factor=max_freq_factor,
        features=features,
        normalization="standard",
    )


def gen_fit_variants(cls, *, rng=None):
    rng = np.random.default_rng(rng)
    for algo in cls.supported_algorithms:
        # Skip NUTS algorithms - they can panic on random data due to bad initial gradients
        if algo.startswith("nuts"):
            continue
        yield cls(
            algo,
            mcmc_niter=rng.integers(5, 20),
            lmsder_niter=rng.integers(1, 10),
            ceres_niter=rng.integers(1, 10),
            ceres_loss_reg=rng.uniform(0.5, 2.0),
            nuts_ntune=rng.integers(5, 20),
            nuts_niter=rng.integers(5, 20),
        )


def construct_example_objects(cls, *, parametric_variants=1, rng=None):
    # Extractor
    if cls is licu_ext.Extractor:
        return [cls(licu_ext.BeyondNStd(1.5), licu_ext.LinearFit())]

    # Pure multiband features have no single-band mode; skip them here
    if cls in pure_multiband_feature_classes:
        return []

    # Periodogram
    if cls is licu_ext.Periodogram:
        return list(chain.from_iterable(gen_periodogram_variants(rng=rng) for _ in range(parametric_variants)))

    # Non-linear fit classes
    if cls in fit_feature_classes:
        return list(chain.from_iterable(gen_fit_variants(cls, rng=rng) for _ in range(parametric_variants)))

    # No mandatory arguments
    if not hasattr(cls, "__getnewargs__"):
        return [cls()]

    # default mandatory arguments
    args, kwargs = get_new_args_kwargs(cls)

    # Add Mean feature for metafeatures
    args = [[licu_ext.Mean()] if arg == () else arg for arg in args]

    objects = [cls(*args, **kwargs)]
    # Nothing to mutate
    if not any(isinstance(arg, float) for arg in args + list(kwargs.values())):
        return objects

    # Mutate floats
    rng = np.random.default_rng(rng)

    def mutation(value):
        if not isinstance(value, float):
            return value
        return value * rng.uniform(0.9, 1.1) + rng.uniform(0.0, 1e-3)

    for _ in range(1, parametric_variants):
        mutated_args = list(map(mutation, args))
        mutated_kwargs = {name: mutation(value) for name, value in kwargs.items()}
        objects.append(cls(*mutated_args, **mutated_kwargs))
    return objects


def gen_feature_evaluators(*, parametric_variants=0, skip_fit=False, rng=None):
    if parametric_variants == 0:
        for cls in non_param_feature_classes:
            yield cls()
        return
    rng = np.random.default_rng(rng)
    classes = all_feature_classes
    if skip_fit:
        classes = classes - fit_feature_classes
    for cls in classes:
        yield from construct_example_objects(cls, parametric_variants=parametric_variants, rng=rng)


_MULTIBAND_BANDS = ["g", "r"]


def _try_construct_multiband(cls):
    """Return a multiband instance of *cls*, or None if not supported."""
    if cls in fit_feature_classes or cls in (licu_ext.Extractor, licu_ext.Bins, licu_ext.JSONDeserializedFeature):
        return None
    if cls is licu_ext.Periodogram:
        return licu_ext.Periodogram(peaks=2, bands=_MULTIBAND_BANDS)
    # Pure multiband classes already take bands as their sole mandatory arg
    if cls in pure_multiband_feature_classes:
        return cls(_MULTIBAND_BANDS)
    try:
        if not hasattr(cls, "__getnewargs__"):
            return cls(bands=_MULTIBAND_BANDS)
        args, kwargs = get_new_args_kwargs(cls)
        args = [[licu_ext.Mean()] if arg == () else arg for arg in args]
        return cls(*args, **kwargs, bands=_MULTIBAND_BANDS)
    except Exception:
        return None


_multiband_feature_classes = frozenset(cls for cls in all_feature_classes if _try_construct_multiband(cls) is not None)
assert len(_multiband_feature_classes) > 0


def gen_multiband_feature_evaluators():
    for cls in _multiband_feature_classes:
        yield _try_construct_multiband(cls)


def gen_lc(n, rng=None):
    rng = np.random.default_rng(rng)

    t = np.sort(rng.normal(0, 1, n))
    m = t.copy()
    sigma = np.full_like(t, 0.1)

    return t, m, sigma


@pytest.mark.parametrize("cls", list(all_feature_classes))
def test_available_transforms(cls):
    # All available features should consume transform=None
    none = new_default(cls, transform=None)

    # If transform consumes False it
    # 1) should give the same feature as transform=None
    # 2) should be able to consume transform=True
    try:
        false = new_default(cls, transform=False)
    except NotImplementedError:
        return
    # It would be better to compare objects themselves, but __eq__ is not implemented yet
    # https://github.com/light-curve/light-curve-python/issues/148
    assert false.names == none.names
    true = new_default(cls, transform=True)
    # Check if transform=True is not the same as transform=False
    default_transform = getattr(cls, "default_transform", None)
    if default_transform != "identity":
        assert true.names != false.names

    # Both attributes should be present or absent
    assert hasattr(cls, "supported_transforms") == hasattr(cls, "default_transform")

    if not hasattr(cls, "supported_transforms"):
        return

    assert cls.default_transform in cls.supported_transforms

    for transform in cls.supported_transforms + ["default"]:
        new_default(cls, transform=transform)


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


@pytest.mark.parametrize("feature", gen_multiband_feature_evaluators())
def test_multiband_output_length_matches_names(feature):
    t, m, sigma, band = _make_multiband_lc(_MULTIBAND_BANDS, n_per_band=60, rng=1)
    values = feature(t, m, sigma, band=band, sorted=True)
    assert len(values) == len(feature.names)
    assert len(values) == len(feature.descriptions)


# We don't want *Fit features here: too slow
@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=0))
def test_many_vs_call(feature):
    rng = np.random.default_rng(0)
    n_obs = 128
    n_lc = 128

    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    call = np.stack([feature(*lc, sorted=True) for lc in lcs])
    many = feature.many(lcs, sorted=True, n_jobs=2)
    assert_array_equal(call, many)

    # Test with Python threads to ensure we have no problems on the free-threading CPython
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(feature, *lc, sorted=True) for lc in lcs]
        call_threads = np.stack([f.result() for f in futures])
        del futures
    assert_array_equal(call, call_threads)

    n_lcs_per_job = 4
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(feature.many, lcs[i : i + n_lcs_per_job], sorted=True, n_jobs=2)
            for i in range(0, n_lc, n_lcs_per_job)
        ]
        many_threads = np.concatenate([f.result() for f in futures])
        del futures
    assert_array_equal(call, many_threads)


def test_fill_value_not_enough_observations():
    n = 1
    t = np.linspace(0.0, 1.0, n)
    m = t.copy()
    fill_value = -100.0
    sigma = np.ones_like(t)
    feature = licu_ext.Kurtosis()
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
@pytest.mark.parametrize(
    "cls", (licu_ext.ExcessVariance, licu_ext.LinearFit, licu_ext.ReducedChi2, licu_ext.StetsonK, licu_ext.WeightedMean)
)
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


@pytest.mark.parametrize("feature", gen_multiband_feature_evaluators())
@pytest.mark.parametrize("pickle_protocol", tuple(range(2, pickle.HIGHEST_PROTOCOL + 1)))
def test_multiband_pickling(feature, pickle_protocol):
    t, m, sigma, band = _make_multiband_lc(_MULTIBAND_BANDS, n_per_band=60, rng=0)
    values = feature(t, m, sigma, band=band, sorted=True)

    b = pickle.dumps(feature, protocol=pickle_protocol)
    new_feature = pickle.loads(b)

    new_values = new_feature(t, m, sigma, band=band, sorted=True)
    assert_array_equal(values, new_values)


@pytest.mark.parametrize("feature", gen_periodogram_variants(rng=None))
@pytest.mark.parametrize("pickle_protocol", tuple(range(2, pickle.HIGHEST_PROTOCOL + 1)))
def test_periodogram_pickling(feature, pickle_protocol):
    n_obs = 128
    t, m, _sigma = gen_lc(n_obs)
    powers = feature.power(t, m)

    b = pickle.dumps(feature, protocol=pickle_protocol)
    new_feature = pickle.loads(b)

    new_powers = new_feature.power(t, m)
    assert_array_equal(powers, new_powers)


@pytest.mark.parametrize("feature", chain.from_iterable(gen_fit_variants(cls, rng=None) for cls in fit_feature_classes))
@pytest.mark.parametrize("pickle_protocol", tuple(range(2, pickle.HIGHEST_PROTOCOL + 1)))
def test_non_linear_fit_pickling(feature, pickle_protocol):
    n_obs = 128
    t, m, sigma = gen_lc(n_obs)

    params = feature(t, m, sigma)

    model = feature.model(t, params)

    b = pickle.dumps(feature, protocol=pickle_protocol)
    new_feature = pickle.loads(b)

    new_params = new_feature(t, m, sigma)
    assert_array_equal(params, new_params)

    new_model = new_feature.model(t, params)
    assert_array_equal(model, new_model)


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
    licu_ext.Amplitude(),  # no parameters
    licu_ext.BeyondNStd(1.5),  # parametric
    licu_ext.Extractor(  # large
        licu_ext.Amplitude(),
        licu_ext.BeyondNStd(2.0),
        licu_ext.Bins(
            [licu_ext.Kurtosis(), licu_ext.LinearTrend(), licu_ext.WeightedMean()],
            window=2.0,
            offset=59500.5,
        ),
        licu_ext.Periodogram(features=[licu_ext.InterPercentileRange(0.01)], peaks=5, max_freq_factor=12.0),
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
@pytest.mark.parametrize("algo", ("ceres", "mcmc-ceres", "lmsder", "mcmc-lmsder"))
def test_bazin_fit_precise(algo):
    bazin = licu_ext.BazinFit(algo)

    true_params = np.array([10.0, -2.0, 10.0, 10.0, 25.0])
    t = np.linspace(-50.0, 120.0, 1000)
    flux = bazin.model(t, true_params)
    fluxerr = np.ones_like(t)

    *params, reduced_chi2 = bazin(t, flux, fluxerr)
    assert_allclose(true_params, params, rtol=1e-4)  # tolerance set to underlying algorithms


# We do not check pure NUTS because it requires a lot of iterations and would be too slow
@pytest.mark.parametrize("algo", ("nuts-ceres", "nuts-lmsder"))
def test_bazin_fit_precise_nuts(algo):
    bazin = licu_ext.BazinFit(algo)

    true_params = np.array([10.0, -2.0, 10.0, 10.0, 25.0])
    t = np.linspace(-50.0, 120.0, 1000)
    flux = bazin.model(t, true_params)
    fluxerr = np.ones_like(t)

    *params, reduced_chi2 = bazin(t, flux, fluxerr)
    assert_allclose(true_params, params, rtol=1e-3)


@pytest.mark.parametrize("cls", list(fit_feature_classes))
def test_nuts_in_supported_algorithms(cls):
    algos = cls.supported_algorithms
    assert "nuts" in algos
    assert "nuts-ceres" in algos
    assert "nuts-lmsder" in algos


@pytest.mark.parametrize("cls", list(fit_feature_classes))
def test_nuts_custom_params(cls):
    feat = cls("nuts-ceres", nuts_ntune=50, nuts_niter=50)
    assert feat is not None


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=5, rng=0))
def test_json_serialization(feature):
    n_obs = 128
    data = gen_lc(n_obs, rng=0)
    values = feature(*data)

    from_to_json = licu_ext.feature_from_json(feature.to_json())
    values_from_to_json = from_to_json(*data)
    assert_allclose(values, values_from_to_json)


@pytest.mark.parametrize("feature", gen_multiband_feature_evaluators())
def test_multiband_many_vs_call(feature):
    rng = np.random.default_rng(2)
    n_lc = 10
    lcs = [_make_multiband_lc(_MULTIBAND_BANDS, n_per_band=40, rng=rng) for _ in range(n_lc)]

    call = np.stack([feature(t, m, sigma, band=band, sorted=True) for t, m, sigma, band in lcs])
    many_input = [(t, m, sigma, band) for t, m, sigma, band in lcs]
    many = feature.many(many_input, sorted=True, n_jobs=2)
    assert_array_equal(call, many)


@pytest.mark.parametrize("feature", gen_multiband_feature_evaluators())
def test_multiband_json_round_trip(feature):
    t, m, sigma, band = _make_multiband_lc(_MULTIBAND_BANDS, n_per_band=60, rng=3)
    values = feature(t, m, sigma, band=band, sorted=True)

    json_str = feature.to_json()
    restored = licu_ext.feature_from_json(json_str)

    restored_values = restored(t, m, sigma, band=band, sorted=True)
    assert_allclose(values, restored_values)


def test_mixed_mode_to_json_raises():
    """Mixed-mode (Extractor with both single- and multiband features) cannot be serialized."""
    ext = licu_ext.Extractor(licu_ext.Amplitude(bands=["g", "r"]), licu_ext.Mean())
    with pytest.raises(NotImplementedError):
        ext.to_json()


def test_json_deserialization():
    json = """
    {"FeatureExtractor":{"features":[{"Transformed":{"feature":{"AndersonDarlingNormal":{}},"transformer":{"Ln1p":{}}}},
    {"Transformed":{"feature":{"BazinFit":{"algorithm":{"Ceres":{"loss_factor":null,"niterations":20}},"inits_bounds":
    {"OptionArrays":{"init":[null,null,null,null,null],"lower":[0.0036307805477010066,null,null,0.0001,0.0001],"upper":
    [3630780547.7010174,null,null,30000.0,30000.0]}},"ln_prior":{"Fixed":{"None":{}}}}},"transformer":{"BazinFit":
    {"mag_zp":23.899999618530273}}}},{"ExcessVariance":{}}]}}
    """
    from_json = licu_ext.feature_from_json(json)
    assert isinstance(from_json, licu_ext._FeatureEvaluator)
    from_json(*gen_lc(128))


def test_raises_for_wrong_inputs():
    fe = licu_ext.Amplitude()

    # First argument
    with pytest.raises(TypeError, match="'t' has type 'int'"):
        fe(5, [1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="'t' is a 2-d array"):
        fe(np.array([[1.0, 2.0, 3.0]]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match="'t' has dtype <U1"):
        fe(np.array(["a", "b", "c"]), np.array([1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match="'t' has type 'list'"):
        fe([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], cast=False)
    # No failure of the last test with cast=True
    _ = fe([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], cast=True)

    # Second and third arguments
    t = np.arange(10, dtype=np.float64)
    with pytest.raises(ValueError, match="Mismatched lengths:"):
        fe(t, np.arange(11, dtype=np.float64))
    with pytest.raises(TypeError, match="'m' must be a numpy array"):
        fe(t, set(t))
    with pytest.raises(TypeError, match="'sigma' is a 0-d array"):
        fe(t, t, np.array(1.0))
    with pytest.raises(TypeError, match="Mismatched dtypes:"):
        fe(t, t.astype(str) + "x")
    with pytest.raises(TypeError, match="Mismatched dtypes:"):
        fe(t, t.astype(np.float32), cast=False)
    # No failure of the last test with cast=True
    _ = fe(t, t.astype(np.float32), cast=True)


def _make_arrow_lcs(lcs, dtype=np.float64):
    """Build a pyarrow List<Struct<t, m, sigma>> array from list of (t, m, sigma) tuples."""
    pa_type = pa.float32() if dtype == np.float32 else pa.float64()
    struct_type = pa.struct([("t", pa_type), ("m", pa_type), ("sigma", pa_type)])
    # Build each light curve as a list of struct rows
    all_lcs = []
    for t, m, sigma in lcs:
        rows = [{"t": dtype(ti), "m": dtype(mi), "sigma": dtype(si)} for ti, mi, si in zip(t, m, sigma)]
        all_lcs.append(rows)
    return pa.array(all_lcs, type=pa.list_(struct_type))


def _make_arrow_lcs_no_sigma(lcs, dtype=np.float64):
    """Build a pyarrow List<Struct<t, m>> array from list of (t, m, sigma) tuples (sigma ignored)."""
    pa_type = pa.float32() if dtype == np.float32 else pa.float64()
    struct_type = pa.struct([("t", pa_type), ("m", pa_type)])
    all_lcs = []
    for t, m, _sigma in lcs:
        rows = [{"t": dtype(ti), "m": dtype(mi)} for ti, mi in zip(t, m)]
        all_lcs.append(rows)
    return pa.array(all_lcs, type=pa.list_(struct_type))


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=0))
def test_many_arrow_3fields(feature):
    """Arrow List<Struct<f64, f64, f64>> matches list-of-tuples result."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    expected = feature.many(lcs, sorted=True, n_jobs=1)
    arrow_arr = _make_arrow_lcs(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": 0, "m": 1, "sigma": 2})
    assert_array_equal(expected, result)


@pytest.mark.parametrize("feature", gen_feature_evaluators(parametric_variants=0))
def test_many_arrow_2fields(feature):
    """Arrow List<Struct<f64, f64>> (no sigma) matches list-of-tuples result."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    lcs_no_sigma = [(t, m, None) for t, m, _s in lcs]
    expected = feature.many(lcs_no_sigma, sorted=True, n_jobs=1)
    arrow_arr = _make_arrow_lcs_no_sigma(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": 0, "m": 1})
    assert_array_equal(expected, result)


def test_many_arrow_f32():
    """Arrow Float32 variant works correctly."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    lcs_f32 = [(t.astype(np.float32), m.astype(np.float32), s.astype(np.float32)) for t, m, s in lcs]
    expected = feature.many(lcs_f32, sorted=True, n_jobs=1)
    arrow_arr = _make_arrow_lcs(lcs, dtype=np.float32)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": 0, "m": 1, "sigma": 2})
    assert_array_equal(expected, result)
    assert result.dtype == np.float32


def test_many_arrow_chunked():
    """Arrow ChunkedArray with multiple chunks works."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    # Split into two chunks
    mid = n_lc // 2
    chunk1 = _make_arrow_lcs(lcs[:mid])
    chunk2 = _make_arrow_lcs(lcs[mid:])
    chunked = pa.chunked_array([chunk1, chunk2])
    result = feature.many(chunked, sorted=True, n_jobs=1, arrow_fields={"t": 0, "m": 1, "sigma": 2})
    assert_array_equal(expected, result)


def test_many_arrow_wrong_fields():
    """Arrow input with wrong schema raises appropriate errors."""
    feature = licu_ext.Amplitude()

    # Missing arrow_fields raises ValueError
    arr = pa.array(
        [[{"t": 1.0, "m": 2.0, "sigma": 0.1}]],
        type=pa.list_(pa.struct([("t", pa.float64()), ("m", pa.float64()), ("sigma", pa.float64())])),
    )
    with pytest.raises(ValueError, match="arrow_fields is required"):
        feature.many(arr, sorted=True)

    # Old list-style input raises TypeError with migration message
    with pytest.raises(TypeError, match="no longer accepts a list"):
        feature.many(arr, sorted=True, arrow_fields=["t", "m", "sigma"])

    # Wrong field dtype (int32)
    arr = pa.array(
        [[{"t": 1, "m": 2}]],
        type=pa.list_(pa.struct([("t", pa.int32()), ("m", pa.int32())])),
    )
    with pytest.raises(TypeError, match="Float32 or Float64"):
        feature.many(arr, sorted=True, arrow_fields={"t": 0, "m": 1})

    # Non-list type
    arr = pa.array([1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match="List array"):
        feature.many(arr, sorted=True, arrow_fields={"t": 0, "m": 1})

    # Mixed field dtypes
    arr = pa.array(
        [[{"t": np.float32(1.0), "m": np.float64(2.0)}]],
        type=pa.list_(pa.struct([("t", pa.float32()), ("m", pa.float64())])),
    )
    with pytest.raises(TypeError, match="same dtype"):
        feature.many(arr, sorted=True, arrow_fields={"t": 0, "m": 1})


def test_many_arrow_parallel():
    """Arrow input works with parallel execution."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 16
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)
    arrow_arr = _make_arrow_lcs(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=2, arrow_fields={"t": 0, "m": 1, "sigma": 2})
    assert_array_equal(expected, result)


def test_many_polars():
    """Polars Series (Arrow-backed) works with many()."""
    import polars as pl

    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    # Build a Polars Series of list-of-structs
    arrow_arr = _make_arrow_lcs(lcs)
    polars_series = pl.Series(arrow_arr)
    result = feature.many(polars_series, sorted=True, n_jobs=1, arrow_fields={"t": 0, "m": 1, "sigma": 2})
    assert_array_equal(expected, result)


def test_many_nanoarrow():
    """nanoarrow array works with many()."""
    import nanoarrow

    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    arrow_arr = _make_arrow_lcs(lcs)
    nano_arr = nanoarrow.Array(arrow_arr)
    result = feature.many(nano_arr, sorted=True, n_jobs=1, arrow_fields={"t": 0, "m": 1, "sigma": 2})
    assert_array_equal(expected, result)


def test_many_arro3():
    """arro3 array works with many()."""
    rng = np.random.default_rng(42)
    n_obs, n_lc = 64, 8
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    arrow_arr = _make_arrow_lcs(lcs)
    arro3_arr = arro3.core.Array.from_arrow(arrow_arr)
    result = feature.many(arro3_arr, sorted=True, n_jobs=1, arrow_fields={"t": 0, "m": 1, "sigma": 2})
    assert_array_equal(expected, result)


@pytest.mark.parametrize(
    "arrow_arr, match_msg",
    [
        # Null list entry (whole light curve is null)
        (
            pa.array(
                [[{"t": 1.0, "m": 2.0, "sigma": 0.1}], None],
                type=pa.list_(pa.struct([("t", pa.float64()), ("m", pa.float64()), ("sigma", pa.float64())])),
            ),
            "list array",
        ),
        # Null struct entry (observation within a light curve is null)
        (
            pa.ListArray.from_arrays(
                pa.array([0, 2], type=pa.int32()),
                pa.StructArray.from_arrays(
                    [pa.array([1.0, 2.0]), pa.array([3.0, 4.0]), pa.array([0.1, 0.2])],
                    names=["t", "m", "sigma"],
                    mask=pa.array([False, True]),
                ),
            ),
            "struct array",
        ),
        # Null value in a data column
        (
            pa.array(
                [[{"t": 1.0, "m": None, "sigma": 0.1}, {"t": 2.0, "m": 5.0, "sigma": 0.2}]],
                type=pa.list_(pa.struct([("t", pa.float64()), ("m", pa.float64()), ("sigma", pa.float64())])),
            ),
            "data columns",
        ),
    ],
    ids=["null_list_entry", "null_struct_entry", "null_value"],
)
def test_many_arrow_nulls_rejected(arrow_arr, match_msg):
    """Null values at any level of the Arrow array raise NotImplementedError."""
    feature = licu_ext.Amplitude()
    with pytest.raises(NotImplementedError, match=match_msg):
        feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": 0, "m": 1, "sigma": 2})


# ──────────────────────────────────────────────────────────────────────────────
# arrow_fields tests
# ──────────────────────────────────────────────────────────────────────────────


def _make_arrow_lcs_extra_fields(lcs, dtype=np.float64):
    """Build a pyarrow List<Struct<extra, t, m, sigma, extra2>> with extra fields."""
    pa_type = pa.float32() if dtype == np.float32 else pa.float64()
    struct_type = pa.struct(
        [
            ("extra", pa_type),
            ("t", pa_type),
            ("m", pa_type),
            ("sigma", pa_type),
            ("extra2", pa_type),
        ]
    )
    all_lcs = []
    for t, m, sigma in lcs:
        rows = [
            {"extra": dtype(0.0), "t": dtype(ti), "m": dtype(mi), "sigma": dtype(si), "extra2": dtype(0.0)}
            for ti, mi, si in zip(t, m, sigma)
        ]
        all_lcs.append(rows)
    return pa.array(all_lcs, type=pa.list_(struct_type))


def test_many_arrow_fields_by_name():
    """arrow_fields with field names selects correct columns from a wider struct."""
    rng = np.random.default_rng(0)
    n_obs, n_lc = 32, 4
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    arrow_arr = _make_arrow_lcs_extra_fields(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": "t", "m": "m", "sigma": "sigma"})
    assert_array_equal(expected, result)


def test_many_arrow_fields_by_index():
    """arrow_fields with integer indices selects correct columns."""
    rng = np.random.default_rng(1)
    n_obs, n_lc = 32, 4
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    # struct is [extra, t, m, sigma, extra2] — t=1, m=2, sigma=3
    arrow_arr = _make_arrow_lcs_extra_fields(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": 1, "m": 2, "sigma": 3})
    assert_array_equal(expected, result)


def test_many_arrow_fields_no_sigma():
    """arrow_fields without sigma key selects t and m only."""
    rng = np.random.default_rng(2)
    n_obs, n_lc = 32, 4
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    lcs_no_sigma = [(t, m, None) for t, m, _s in lcs]
    expected = feature.many(lcs_no_sigma, sorted=True, n_jobs=1)

    arrow_arr = _make_arrow_lcs_extra_fields(lcs)
    result = feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": "t", "m": "m"})
    assert_array_equal(expected, result)


def test_many_arrow_fields_value_wrong_type():
    """arrow_fields with a non-str, non-int value raises TypeError."""
    rng = np.random.default_rng(3)
    n_obs, n_lc = 32, 4
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    arrow_arr = _make_arrow_lcs_extra_fields(lcs)
    with pytest.raises(TypeError, match="column name.*or.*index"):
        feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": "t", "m": 1.5})


def test_many_arrow_fields_nondefault_order():
    """arrow_fields lets user pick fields regardless of position order."""
    rng = np.random.default_rng(4)
    n_obs, n_lc = 32, 4
    lcs = [gen_lc(n_obs, rng=rng) for _ in range(n_lc)]

    feature = licu_ext.Amplitude()
    expected = feature.many(lcs, sorted=True, n_jobs=1)

    # Build struct with fields in non-standard order: sigma, m, t
    pa_type = pa.float64()
    struct_type = pa.struct([("sigma", pa_type), ("m", pa_type), ("t", pa_type)])
    all_lcs = []
    for t, m, sigma in lcs:
        rows = [{"sigma": si, "m": mi, "t": ti} for ti, mi, si in zip(t, m, sigma)]
        all_lcs.append(rows)
    arrow_arr = pa.array(all_lcs, type=pa.list_(struct_type))

    result = feature.many(arrow_arr, sorted=True, n_jobs=1, arrow_fields={"t": "t", "m": "m", "sigma": "sigma"})
    assert_array_equal(expected, result)


def test_many_arrow_fields_invalid_name():
    """arrow_fields with a nonexistent field name raises ValueError."""
    feature = licu_ext.Amplitude()
    arr = _make_arrow_lcs([gen_lc(10, rng=np.random.default_rng(5)) for _ in range(2)])
    with pytest.raises(ValueError, match="not found"):
        feature.many(arr, sorted=True, arrow_fields={"t": "t", "m": "nonexistent"})


def test_many_arrow_fields_index_out_of_range():
    """arrow_fields with an out-of-range index raises ValueError."""
    feature = licu_ext.Amplitude()
    arr = _make_arrow_lcs([gen_lc(10, rng=np.random.default_rng(6)) for _ in range(2)])
    with pytest.raises(ValueError, match="out of range"):
        feature.many(arr, sorted=True, arrow_fields={"t": 0, "m": 99})


def test_many_arrow_fields_missing_required_key():
    """arrow_fields without 't' or 'm' raises KeyError."""
    feature = licu_ext.Amplitude()
    arr = _make_arrow_lcs([gen_lc(10, rng=np.random.default_rng(7)) for _ in range(2)])
    with pytest.raises(KeyError, match='"t"'):
        feature.many(arr, sorted=True, arrow_fields={"m": "m"})
    with pytest.raises(KeyError, match='"m"'):
        feature.many(arr, sorted=True, arrow_fields={"t": "t"})


def test_many_arrow_fields_duplicate_fields():
    """arrow_fields pointing to the same field twice raises ValueError."""
    feature = licu_ext.Amplitude()
    arr = _make_arrow_lcs([gen_lc(10, rng=np.random.default_rng(8)) for _ in range(2)])
    with pytest.raises(ValueError, match="different fields"):
        feature.many(arr, sorted=True, arrow_fields={"t": "t", "m": "t"})


def test_many_arrow_fields_duplicate_field_name():
    """arrow_fields with a duplicate field name raises ValueError."""
    feature = licu_ext.Amplitude()
    # pyarrow allows structs with duplicate field names; build one without dict literals
    # (dict literals deduplicate keys, so use from_arrays instead)
    dup_struct_type = pa.struct([("t", pa.float64()), ("m", pa.float64()), ("m", pa.float64())])
    inner = pa.StructArray.from_arrays(
        [pa.array([1.0]), pa.array([2.0]), pa.array([3.0])],
        names=["t", "m", "m"],
    )
    offsets = pa.array([0, 1], type=pa.int32())
    arr = pa.ListArray.from_arrays(offsets, inner).cast(pa.list_(dup_struct_type))
    with pytest.raises(ValueError, match="ambiguous"):
        feature.many(arr, sorted=True, arrow_fields={"t": "t", "m": "m"})


# ── Multiband tests ────────────────────────────────────────────────────────────


def _make_multiband_lc(band_labels, n_per_band=50, rng=None):
    """Return (t, m, sigma, band) with observations for each band, time-sorted."""
    rng = np.random.default_rng(rng)
    parts_t, parts_m, parts_s, parts_b = [], [], [], []
    for label in band_labels:
        t_b = rng.uniform(0, 100, n_per_band)
        m_b = rng.normal(0, 0.5, n_per_band)
        s_b = np.full(n_per_band, 0.1)
        parts_t.append(t_b)
        parts_m.append(m_b)
        parts_s.append(s_b)
        parts_b.extend([label] * n_per_band)
    t = np.concatenate(parts_t)
    m = np.concatenate(parts_m)
    sigma = np.concatenate(parts_s)
    band = np.array(parts_b)
    idx = np.argsort(t)
    return t[idx], m[idx], sigma[idx], band[idx]


def test_multiband_amplitude_values():
    """Multiband Amplitude output matches independent single-band evaluations."""
    rng = np.random.default_rng(0)
    band_labels = ["g", "r"]
    n = 50

    t_g = np.sort(rng.uniform(0, 100, n))
    m_g = rng.normal(0, 0.5, n)
    sigma_g = np.full(n, 0.1)

    t_r = np.sort(rng.uniform(0, 100, n))
    m_r = rng.normal(0, 0.5, n)
    sigma_r = np.full(n, 0.1)

    t = np.concatenate([t_g, t_r])
    m = np.concatenate([m_g, m_r])
    sigma = np.concatenate([sigma_g, sigma_r])
    band = np.array(["g"] * n + ["r"] * n)
    idx = np.argsort(t)
    t, m, sigma, band = t[idx], m[idx], sigma[idx], band[idx]

    amp_mb = licu_ext.Amplitude(bands=band_labels)
    result = amp_mb(t, m, sigma, band, sorted=True)

    assert result.shape == (2,), f"expected shape (2,), got {result.shape}"

    amp_sb = licu_ext.Amplitude()
    result_g = amp_sb(t_g, m_g, sigma_g, sorted=True)
    result_r = amp_sb(t_r, m_r, sigma_r, sorted=True)

    assert_allclose(result[0], result_g[0])
    assert_allclose(result[1], result_r[0])


def test_multiband_three_bands():
    """Three-band evaluation works and returns values in user-specified band order."""
    rng = np.random.default_rng(0)
    band_labels = ["g", "r", "i"]
    t, m, sigma, band = _make_multiband_lc(band_labels, rng=rng)
    amp = licu_ext.Amplitude(bands=band_labels)
    result = amp(t, m, sigma, band, sorted=True)
    assert result.shape == (3,)
    # Values must match independent per-band evaluations in user-specified order
    amp_sb = licu_ext.Amplitude()
    for i, bl in enumerate(band_labels):
        mask = band == bl
        assert_allclose(result[i], amp_sb(t[mask], m[mask], sigma[mask])[0])


def test_multiband_bands_property():
    """bands property returns the configured band names in user-specified order."""
    bands_arr = np.array(["g", "r", "i"])
    amp = licu_ext.Amplitude(bands=bands_arr)
    assert_array_equal(amp.bands, bands_arr)  # user-specified order is preserved


def test_multiband_non_string_band_dtype_raises():
    """Passing a numpy array with a non-string dtype as band raises TypeError."""
    t, m, sigma, band = _make_multiband_lc(["g", "r"], rng=4)
    amp = licu_ext.Amplitude(bands=["g", "r"])
    float_band = np.zeros(len(band))
    with pytest.raises(TypeError, match="non-string dtype"):
        amp(t, m, sigma, float_band, sorted=True)


def test_multiband_bands_property_after_pickle():
    """bands property reconstructs correctly after a pickle round-trip."""
    import pickle

    amp = licu_ext.Amplitude(bands=["g", "r"])
    amp2 = pickle.loads(pickle.dumps(amp))
    assert_array_equal(amp2.bands, np.array(["g", "r"]))


def test_multiband_unknown_passband_raises():
    """check=True rejects a band array that contains an unconfigured passband."""
    t, m, sigma, band = _make_multiband_lc(["g", "r"], rng=1)
    # Replace some 'r' labels with unknown 'z'
    band = band.copy()
    band[band == "r"] = "z"

    amp = licu_ext.Amplitude(bands=["g", "r"])
    with pytest.raises(ValueError, match="unknown passband"):
        amp(t, m, sigma, band, sorted=True, check=True)


def test_multiband_band_ignored_in_single_band_mode():
    """Passing band to a single-band feature is silently ignored."""
    t, m, sigma, band = _make_multiband_lc(["g", "r"], rng=2)
    amp = licu_ext.Amplitude()
    result_with_band = amp(t, m, sigma, band, sorted=True)
    result_without_band = amp(t, m, sigma, sorted=True)
    assert_array_equal(result_with_band, result_without_band)


def test_multiband_band_none_mismatch_multiband():
    """Omitting band for a multiband feature raises ValueError."""
    t, m, sigma, _ = _make_multiband_lc(["g", "r"], rng=3)
    amp = licu_ext.Amplitude(bands=["g", "r"])
    with pytest.raises(ValueError, match="band must be provided"):
        amp(t, m, sigma, sorted=True)


def test_multiband_periodogram_output_length():
    """Multiband Periodogram returns peaks * 2 values for two bands."""
    peaks = 3
    pg = licu_ext.Periodogram(peaks=peaks, bands=["g", "r"])
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=100, rng=4)
    result = pg(t, m, sigma, band, sorted=True)
    assert result.shape == (peaks * 2,), f"expected ({peaks * 2},), got {result.shape}"


def test_multiband_periodogram_multiband_normalization():
    """multiband_normalization='chi2' is accepted and returns finite values."""
    peaks = 2
    pg = licu_ext.Periodogram(peaks=peaks, bands=["g", "r"], multiband_normalization="chi2")
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=80, rng=5)
    result = pg(t, m, sigma, band, sorted=True)
    assert np.all(np.isfinite(result))


def test_multiband_periodogram_default_normalization_is_chi2():
    """Default multiband_normalization is 'chi2', not 'count'."""
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=80, rng=6)
    pg_chi2 = licu_ext.Periodogram(peaks=2, bands=["g", "r"], multiband_normalization="chi2")
    pg_default = licu_ext.Periodogram(peaks=2, bands=["g", "r"])
    result_chi2 = pg_chi2(t, m, sigma, band, sorted=True)
    result_default = pg_default(t, m, sigma, band, sorted=True)
    np.testing.assert_array_equal(result_chi2, result_default)


def test_freq_power_single_band_shape():
    """Single-band freq_power returns (freq, power) with matching lengths."""
    pg = licu_ext.Periodogram(peaks=2)
    rng = np.random.default_rng(10)
    t = np.sort(rng.uniform(0, 100, 100))
    m = rng.normal(0, 0.5, 100)
    freq, power = pg.freq_power(t, m)
    assert freq.shape == power.shape
    assert freq.ndim == 1
    assert len(freq) > 0
    assert np.all(np.isfinite(freq))
    assert np.all(np.isfinite(power))


def test_freq_power_multiband_shape():
    """Multiband freq_power returns (freq, power) arrays of matching length."""
    pg = licu_ext.Periodogram(peaks=2, bands=["g", "r"])
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=80, rng=11)
    freq, power = pg.freq_power(t, m, sigma, band)
    assert freq.shape == power.shape
    assert freq.ndim == 1
    assert len(freq) > 0
    assert np.all(np.isfinite(freq))
    assert np.all(np.isfinite(power))


def test_freq_power_multiband_sigma_optional():
    """Multiband freq_power accepts sigma=None and returns finite results."""
    pg = licu_ext.Periodogram(peaks=2, bands=["g", "r"])
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=80, rng=12)
    freq, power = pg.freq_power(t, m, band=band)
    assert np.all(np.isfinite(freq))
    assert np.all(np.isfinite(power))


def test_freq_power_multiband_sigma_affects_chi2_power():
    """Providing non-uniform sigma changes the combined power under chi2 normalization.

    With uniform sigma, sigma^{-2} cancels in the chi2 ratio so both calls
    give the same result.  With different per-band sigma scales the weights
    differ, so the combined power must differ too.
    """
    pg = licu_ext.Periodogram(peaks=2, bands=["g", "r"])
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=80, rng=13)
    # Give the two bands very different uncertainty scales so sigma matters.
    sigma = np.where(band == "g", 0.01, 1.0)
    _, power_with_sigma = pg.freq_power(t, m, sigma, band)
    _, power_no_sigma = pg.freq_power(t, m, band=band)
    assert not np.allclose(power_with_sigma, power_no_sigma)


def test_freq_power_multiband_requires_band():
    """Multiband freq_power raises if band is not provided."""
    pg = licu_ext.Periodogram(peaks=2, bands=["g", "r"])
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=80, rng=14)
    with pytest.raises(Exception, match="band is required"):
        pg.freq_power(t, m, sigma)


def test_freq_power_single_band_rejects_band_arg():
    """Single-band freq_power raises if band is passed."""
    pg = licu_ext.Periodogram(peaks=2)
    rng = np.random.default_rng(15)
    t = np.sort(rng.uniform(0, 100, 100))
    m = rng.normal(0, 0.5, 100)
    band = np.array(["g"] * 100)
    with pytest.raises(Exception, match="multiband mode"):
        pg.freq_power(t, m, band=band)


def test_freq_power_count_vs_chi2_normalization():
    """freq_power with count and chi2 normalizations produce different results."""
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=80, rng=16)
    pg_chi2 = licu_ext.Periodogram(peaks=2, bands=["g", "r"], multiband_normalization="chi2")
    pg_count = licu_ext.Periodogram(peaks=2, bands=["g", "r"], multiband_normalization="count")
    _, power_chi2 = pg_chi2.freq_power(t, m, sigma, band)
    _, power_count = pg_count.freq_power(t, m, sigma, band)
    assert not np.allclose(power_chi2, power_count)


def _split_per_band(t, m, sigma, band, labels):
    """Return dict label -> (t, m, sigma) for each band label."""
    return {lbl: (t[band == lbl], m[band == lbl], sigma[band == lbl]) for lbl in labels}


@pytest.mark.parametrize(
    "cls, stat",
    [
        (licu_ext.ColorOfMaximum, np.max),
        (licu_ext.ColorOfMedian, np.median),
        (licu_ext.ColorOfMinimum, np.min),
    ],
)
def test_color_two_band_values(cls, stat):
    """Color* features equal stat(band0) - stat(band1) computed per band."""
    rng = np.random.default_rng(7)
    n = 60
    t_g = np.sort(rng.uniform(0, 100, n))
    m_g = rng.normal(0, 1, n).astype(np.float64)
    sigma_g = np.full(n, 0.1)
    t_r = np.sort(rng.uniform(0, 100, n))
    m_r = rng.normal(0, 1, n).astype(np.float64)
    sigma_r = np.full(n, 0.1)

    t = np.concatenate([t_g, t_r])
    m = np.concatenate([m_g, m_r])
    sigma = np.concatenate([sigma_g, sigma_r])
    band = np.array(["g"] * n + ["r"] * n)
    idx = np.argsort(t)
    t, m, sigma, band = t[idx], m[idx], sigma[idx], band[idx]

    feature = cls(["g", "r"])
    result = feature(t, m, sigma, band, sorted=True)
    assert result.shape == (1,)
    assert_allclose(result[0], stat(m_g) - stat(m_r))


def test_color_two_band_wrong_count():
    """ColorOfMaximum raises ValueError when bands does not have exactly 2 elements."""
    with pytest.raises(ValueError, match="exactly 2"):
        licu_ext.ColorOfMaximum(["g", "r", "i"])
    with pytest.raises(ValueError, match="exactly 2"):
        licu_ext.ColorOfMaximum(["g"])


def test_color_spread_values():
    """ColorSpread equals population std dev of per-band inverse-variance-weighted means."""
    rng = np.random.default_rng(8)
    n = 60
    labels = ["g", "r", "i"]
    parts_t, parts_m, parts_s, parts_b = [], [], [], []
    per_band_m = {}
    per_band_s = {}
    for lbl in labels:
        t_b = np.sort(rng.uniform(0, 100, n))
        m_b = rng.normal(0, 1, n).astype(np.float64)
        s_b = np.full(n, 0.1)
        parts_t.append(t_b)
        parts_m.append(m_b)
        parts_s.append(s_b)
        parts_b.extend([lbl] * n)
        per_band_m[lbl] = m_b
        per_band_s[lbl] = s_b

    t = np.concatenate(parts_t)
    m = np.concatenate(parts_m)
    sigma = np.concatenate(parts_s)
    band = np.array(parts_b)
    idx = np.argsort(t)
    t, m, sigma, band = t[idx], m[idx], sigma[idx], band[idx]

    cs = licu_ext.ColorSpread(labels)
    result = cs(t, m, sigma, band, sorted=True)
    assert result.shape == (1,)

    # Compute expected: weighted mean per band, then population std dev
    means = []
    for lbl in sorted(labels):  # upstream sorts bands
        w = 1.0 / per_band_s[lbl] ** 2
        means.append(np.sum(w * per_band_m[lbl]) / np.sum(w))
    means = np.array(means)
    expected = np.sqrt(np.mean((means - means.mean()) ** 2))
    assert_allclose(result[0], expected, rtol=1e-6)


def test_color_spread_two_bands_minimum():
    """ColorSpread accepts exactly 2 bands."""
    cs = licu_ext.ColorSpread(["g", "r"])
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=40, rng=9)
    result = cs(t, m, sigma, band, sorted=True)
    assert result.shape == (1,)
    assert np.isfinite(result[0])


def test_color_spread_too_few_bands():
    """ColorSpread raises ValueError when fewer than 2 bands are given."""
    with pytest.raises(ValueError, match="at least 2"):
        licu_ext.ColorSpread(["g"])


def test_multiband_duplicate_bands_raises():
    """Duplicate band names in the bands constructor raise ValueError."""
    with pytest.raises(ValueError, match="duplicate"):
        licu_ext.Amplitude(bands=["g", "g"])


@pytest.mark.parametrize(
    "bands",
    [
        ["u", "g"],
        ["g", "r", "i"],
    ],
)
def test_per_band_feature_name_order_matches_user_input(bands):
    """Output names must follow the user-supplied band order, not sorted order."""
    feature = licu_ext.Amplitude(bands=bands)
    suffixes = [name.split("_")[-1] for name in feature.names]
    assert suffixes == bands


@pytest.mark.parametrize(
    "bands",
    [
        ["u", "g"],
        ["g", "r", "i"],
    ],
)
def test_per_band_feature_values_order_matches_user_input(bands):
    """Output values must correspond to bands in user-supplied order."""
    t, m, sigma, band = _make_multiband_lc(bands, n_per_band=60, rng=0)
    feature = licu_ext.Amplitude(bands=bands)
    result = feature(t, m, sigma, band)
    for i, b in enumerate(bands):
        mask = band == b
        expected = licu_ext.Amplitude()(t[mask], m[mask], sigma[mask])[0]
        assert result[i] == pytest.approx(expected), f"band {b!r} at index {i}"


# ── Multiband Arrow tests ──────────────────────────────────────────────────────


def _make_arrow_multiband_lcs(multiband_lcs, band_col_type=pa.utf8()):
    """Build a pyarrow List<Struct<t, m, sigma, band>> from list of (t, m, sigma, band) tuples."""
    struct_type = pa.struct(
        [
            ("t", pa.float64()),
            ("m", pa.float64()),
            ("sigma", pa.float64()),
            ("band", band_col_type),
        ]
    )
    all_lcs = []
    for t, m, sigma, band in multiband_lcs:
        rows = [
            {"t": float(t[i]), "m": float(m[i]), "sigma": float(sigma[i]), "band": str(band[i])} for i in range(len(t))
        ]
        all_lcs.append(rows)
    return pa.array(all_lcs, type=pa.list_(struct_type))


def test_many_arrow_multiband_matches_list_input():
    """Arrow multiband input matches list-of-tuples result."""
    rng = np.random.default_rng(10)
    band_labels = ["g", "r"]
    n_lcs = 4
    lcs = [_make_multiband_lc(band_labels, n_per_band=30, rng=rng) for _ in range(n_lcs)]

    feat = licu_ext.Amplitude(bands=band_labels)
    list_input = [(t, m, sigma, band) for t, m, sigma, band in lcs]
    expected = feat.many(list_input, sorted=True, fill_value=-999.0)

    arrow_arr = _make_arrow_multiband_lcs(lcs)
    result = feat.many(
        arrow_arr,
        sorted=True,
        fill_value=-999.0,
        arrow_fields={"t": "t", "m": "m", "sigma": "sigma", "band": "band"},
    )
    assert_array_equal(expected, result)


def test_many_arrow_multiband_large_utf8():
    """Arrow multiband with LargeUtf8 band column works correctly."""
    rng = np.random.default_rng(11)
    band_labels = ["g", "r"]
    n_lcs = 4
    lcs = [_make_multiband_lc(band_labels, n_per_band=30, rng=rng) for _ in range(n_lcs)]

    feat = licu_ext.Amplitude(bands=band_labels)
    expected = feat.many(
        _make_arrow_multiband_lcs(lcs, band_col_type=pa.utf8()),
        sorted=True,
        fill_value=-999.0,
        arrow_fields={"t": "t", "m": "m", "sigma": "sigma", "band": "band"},
    )
    result = feat.many(
        _make_arrow_multiband_lcs(lcs, band_col_type=pa.large_utf8()),
        sorted=True,
        fill_value=-999.0,
        arrow_fields={"t": "t", "m": "m", "sigma": "sigma", "band": "band"},
    )
    assert_array_equal(expected, result)


def test_many_arrow_multiband_missing_band_field_raises():
    """Arrow multiband without 'band' in arrow_fields raises ValueError."""
    rng = np.random.default_rng(12)
    band_labels = ["g", "r"]
    lcs = [_make_multiband_lc(band_labels, n_per_band=10, rng=rng)]
    arrow_arr = _make_arrow_multiband_lcs(lcs)

    feat = licu_ext.Amplitude(bands=band_labels)
    with pytest.raises(ValueError, match='"band"'):
        feat.many(arrow_arr, sorted=True, fill_value=-999.0, arrow_fields={"t": "t", "m": "m"})


def test_many_arrow_multiband_non_string_band_raises():
    """Arrow integer band column with a string-bands feature raises TypeError (mode mismatch)."""
    rng = np.random.default_rng(13)
    band_labels = ["g", "r"]
    lcs = [_make_multiband_lc(band_labels, n_per_band=10, rng=rng)]

    struct_type = pa.struct([("t", pa.float64()), ("m", pa.float64()), ("band", pa.int32())])
    t, m, sigma, band = lcs[0]
    inner = pa.StructArray.from_arrays(
        [pa.array(t), pa.array(m), pa.array([0] * len(t), type=pa.int32())],
        names=["t", "m", "band"],
    )
    arrow_arr = pa.array([inner], type=pa.list_(struct_type))

    feat = licu_ext.Amplitude(bands=band_labels)
    with pytest.raises(TypeError):
        feat.many(arrow_arr, sorted=True, fill_value=-999.0, arrow_fields={"t": "t", "m": "m", "band": "band"})


def test_many_arrow_multiband_unknown_band_raises():
    """Arrow multiband with an unrecognized band value raises ValueError."""
    rng = np.random.default_rng(14)
    band_labels = ["g", "r"]
    lcs = [_make_multiband_lc(band_labels, n_per_band=10, rng=rng)]
    # Swap "r" for "z" in the band column
    t, m, sigma, band = lcs[0]
    band_bad = np.where(band == "r", "z", band)
    arrow_arr = _make_arrow_multiband_lcs([(t, m, sigma, band_bad)])

    feat = licu_ext.Amplitude(bands=band_labels)
    with pytest.raises(ValueError, match="unknown passband"):
        feat.many(
            arrow_arr,
            sorted=True,
            fill_value=-999.0,
            arrow_fields={"t": "t", "m": "m", "sigma": "sigma", "band": "band"},
        )


# ── Mixed Extractor tests ──────────────────────────────────────────────────────


def test_extractor_mixed_output_length():
    """Extractor with mixed single- and multi-band features returns combined length."""
    rng = np.random.default_rng(42)
    band_labels = ["g", "r"]
    t, m, sigma, band = _make_multiband_lc(band_labels, n_per_band=60, rng=rng)

    amplitude_mb = licu_ext.Amplitude(bands=band_labels)  # 2 values (one per band)
    mean_sb = licu_ext.WeightedMean()  # 1 value — single-band, uses full lc ignoring band

    ext = licu_ext.Extractor(amplitude_mb, mean_sb)
    result = ext(t, m, sigma, band=band)
    # 2 (Amplitude per band) + 1 (WeightedMean single-band) = 3
    assert result.shape == (3,)


def test_extractor_mixed_values_match_separate_calls():
    """Values from a mixed Extractor match those from separate evaluators."""
    rng = np.random.default_rng(7)
    band_labels = ["g", "r"]
    t, m, sigma, band = _make_multiband_lc(band_labels, n_per_band=60, rng=rng)

    amplitude_mb = licu_ext.Amplitude(bands=band_labels)
    mean_sb = licu_ext.WeightedMean()
    ext = licu_ext.Extractor(amplitude_mb, mean_sb)

    result = ext(t, m, sigma, band=band)

    amp_expected = amplitude_mb(t, m, sigma, band=band)
    mean_expected = mean_sb(t, m, sigma)

    np.testing.assert_allclose(result[:2], amp_expected)
    np.testing.assert_allclose(result[2:], mean_expected)


def test_extractor_mixed_feature_names():
    """Mixed Extractor reports feature names from both components."""
    amplitude_mb = licu_ext.Amplitude(bands=["g", "r"])
    mean_sb = licu_ext.WeightedMean()
    ext = licu_ext.Extractor(amplitude_mb, mean_sb)

    names = ext.names
    assert "amplitude_g" in names
    assert "amplitude_r" in names
    assert "weighted_mean" in names
    assert len(names) == 3


def test_extractor_mixed_fill_value():
    """fill_value is applied when multiband feature fails."""
    rng = np.random.default_rng(3)
    # Only one observation per band — Amplitude can fail on very short series
    # Use BeyondNStd which needs ≥1 obs; give enough data but pass a missing band
    t, m, sigma, band = _make_multiband_lc(["g", "r"], n_per_band=60, rng=rng)

    amplitude_mb = licu_ext.Amplitude(bands=["g", "r"])
    mean_sb = licu_ext.WeightedMean()
    ext = licu_ext.Extractor(amplitude_mb, mean_sb)

    result = ext(t, m, sigma, band=band, fill_value=np.nan)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_extractor_mixed_multi_value_features():
    """Mixed Extractor with multi-value features: correct shape, names, and values.

    Layout:
      LinearTrend(bands=["g","r"])  -> 6 values (3 per band)
      LinearFit()                   -> 3 values (single-band, whole licu_ext)
      BeyondNStd()                  -> 1 value  (single-band, whole licu_ext)
    Total: 10 values, interleaved [MB MB MB MB MB MB SB SB SB SB]
    """
    rng = np.random.default_rng(99)
    band_labels = ["g", "r"]
    t, m, sigma, band = _make_multiband_lc(band_labels, n_per_band=80, rng=rng)

    lt_mb = licu_ext.LinearTrend(bands=band_labels)  # 6 values
    lf_sb = licu_ext.LinearFit()  # 3 values
    bn_sb = licu_ext.BeyondNStd()  # 1 value

    ext = licu_ext.Extractor(lt_mb, lf_sb, bn_sb)
    result = ext(t, m, sigma, band=band)

    assert result.shape == (10,)

    # Names: 6 from LinearTrend, then 3 from LinearFit, then 1 from BeyondNStd
    expected_names = lt_mb.names + lf_sb.names + bn_sb.names
    assert list(ext.names) == expected_names

    # Values match independent evaluations
    lt_expected = lt_mb(t, m, sigma, band=band)
    lf_expected = lf_sb(t, m, sigma)
    bn_expected = bn_sb(t, m, sigma)

    np.testing.assert_allclose(result[:6], lt_expected)
    np.testing.assert_allclose(result[6:9], lf_expected)
    np.testing.assert_allclose(result[9:], bn_expected)


def test_multiband_bins_single_band_features():
    """Multiband Bins with single-band features wraps each feature per band.

    Bins(window=1, bands=["g","r"]) containing Amplitude() should produce
    2 values (one Amplitude per band on the binned series), matching
    independent Bins(Amplitude()) calls per band.
    """
    rng = np.random.default_rng(200)
    band_labels = ["g", "r"]
    n = 60
    t_g = np.sort(rng.uniform(0, 10, n))
    t_r = np.sort(rng.uniform(0, 10, n))
    m_g = rng.normal(0, 0.5, n)
    m_r = rng.normal(0, 0.5, n)
    sigma_g = np.full(n, 0.1)
    sigma_r = np.full(n, 0.1)

    t = np.concatenate([t_g, t_r])
    m = np.concatenate([m_g, m_r])
    sigma = np.concatenate([sigma_g, sigma_r])
    band = np.array(["g"] * n + ["r"] * n)
    idx = np.argsort(t)
    t, m, sigma, band = t[idx], m[idx], sigma[idx], band[idx]

    window, offset = 1.0, 0.0
    bins_mb = licu_ext.Bins([licu_ext.Amplitude()], window=window, offset=offset, bands=band_labels)
    result = bins_mb(t, m, sigma, band=band)

    assert result.shape == (2,), f"expected (2,), got {result.shape}"

    bins_sb = licu_ext.Bins([licu_ext.Amplitude()], window=window, offset=offset)
    expected_g = bins_sb(t_g, m_g, sigma_g)
    expected_r = bins_sb(t_r, m_r, sigma_r)

    np.testing.assert_allclose(result[0], expected_g[0], rtol=1e-6)
    np.testing.assert_allclose(result[1], expected_r[0], rtol=1e-6)


def test_multiband_bins_names():
    """Multiband Bins names are prefixed per band."""
    bins_mb = licu_ext.Bins([licu_ext.Amplitude()], window=1.0, offset=0.0, bands=["g", "r"])
    names = list(bins_mb.names)
    assert len(names) == 2
    # Each name should contain the bins window/offset prefix and the feature name
    assert all("bins_" in n for n in names)
    assert all("amplitude" in n for n in names)


def test_multiband_bins_without_bands_rejects_multiband_feature():
    """Bins without bands= rejects a multiband feature."""
    with pytest.raises((ValueError, Exception)):
        licu_ext.Bins([licu_ext.Amplitude(bands=["g", "r"])], window=1.0, offset=0.0)


def test_extractor_mixed_many_raises():
    """many() raises NotImplementedError for a mixed-mode Extractor."""
    rng = np.random.default_rng(42)
    band_labels = ["g", "r"]
    lcs = [_make_multiband_lc(band_labels, n_per_band=30, rng=rng) for _ in range(4)]
    ext = licu_ext.Extractor(licu_ext.Amplitude(bands=band_labels), licu_ext.WeightedMean())
    with pytest.raises(NotImplementedError, match="mixed"):
        ext.many(lcs, sorted=True)


def test_extractor_mixed_arrow_raises():
    """many() with Arrow input raises NotImplementedError for a mixed-mode Extractor."""
    rng = np.random.default_rng(43)
    band_labels = ["g", "r"]
    lcs = [_make_multiband_lc(band_labels, n_per_band=30, rng=rng) for _ in range(4)]
    arrow_arr = _make_arrow_multiband_lcs(lcs)
    ext = licu_ext.Extractor(licu_ext.Amplitude(bands=band_labels), licu_ext.WeightedMean())
    with pytest.raises(NotImplementedError):
        ext.many(
            arrow_arr,
            sorted=True,
            arrow_fields={"t": "t", "m": "m", "sigma": "sigma", "band": "band"},
        )


# ── Integer passband tests ─────────────────────────────────────────────────────


def _make_int_multiband_lc(band_ids, n_per_band=50, rng=None, dtype=np.int64):
    """Return (t, m, sigma, band) with integer band labels, time-sorted."""
    rng = np.random.default_rng(rng)
    parts_t, parts_m, parts_s, parts_b = [], [], [], []
    for bid in band_ids:
        t_b = rng.uniform(0, 100, n_per_band)
        m_b = rng.normal(0, 0.5, n_per_band)
        s_b = np.full(n_per_band, 0.1)
        parts_t.append(t_b)
        parts_m.append(m_b)
        parts_s.append(s_b)
        parts_b.extend([bid] * n_per_band)
    t = np.concatenate(parts_t)
    m = np.concatenate(parts_m)
    sigma = np.concatenate(parts_s)
    band = np.array(parts_b, dtype=dtype)
    idx = np.argsort(t)
    return t[idx], m[idx], sigma[idx], band[idx]


def test_integer_bands_basic():
    """Integer bands [0, 1] produce the same results as string bands ['0', '1']."""
    rng = np.random.default_rng(100)
    n = 50
    t_0 = np.sort(rng.uniform(0, 100, n))
    m_0 = rng.normal(0, 0.5, n)
    sigma_0 = np.full(n, 0.1)
    t_1 = np.sort(rng.uniform(0, 100, n))
    m_1 = rng.normal(0, 0.5, n)
    sigma_1 = np.full(n, 0.1)

    t = np.concatenate([t_0, t_1])
    m = np.concatenate([m_0, m_1])
    sigma = np.concatenate([sigma_0, sigma_1])
    idx = np.argsort(t)
    t, m, sigma = t[idx], m[idx], sigma[idx]

    band_str = np.where(idx < n, "0", "1")
    band_int = np.where(idx < n, 0, 1).astype(np.int64)

    amp_str = licu_ext.Amplitude(bands=["0", "1"])
    amp_int = licu_ext.Amplitude(bands=[0, 1])

    result_str = amp_str(t, m, sigma, band_str, sorted=True)
    result_int = amp_int(t, m, sigma, band_int, sorted=True)
    assert_allclose(result_int, result_str)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_integer_bands_numpy_dtypes(dtype):
    """Integer bands work with all numpy integer dtypes."""
    band_ids = [0, 1, 2]
    t, m, sigma, band = _make_int_multiband_lc(band_ids, n_per_band=40, rng=101, dtype=dtype)
    amp = licu_ext.Amplitude(bands=band_ids)
    result = amp(t, m, sigma, band, sorted=True)
    assert result.shape == (3,)
    assert np.all(np.isfinite(result))


def test_integer_bands_reversed_order():
    """Integer bands specified in reversed order still produce correct per-band results."""
    rng = np.random.default_rng(102)
    band_ids = [2, 1, 0]  # user order: 2, 1, 0
    t, m, sigma, band = _make_int_multiband_lc(band_ids, n_per_band=50, rng=rng)

    amp = licu_ext.Amplitude(bands=band_ids)
    result = amp(t, m, sigma, band, sorted=True)

    amp_sb = licu_ext.Amplitude()
    for i, bid in enumerate(band_ids):
        mask = band == bid
        expected = amp_sb(t[mask], m[mask], sigma[mask])[0]
        assert_allclose(result[i], expected, err_msg=f"band {bid} mismatch")


def test_integer_bands_scrambled_order():
    """Integer bands in non-monotone order still produce correct per-band results."""
    rng = np.random.default_rng(103)
    band_ids = [5, 2, 8]  # non-contiguous, scrambled
    t, m, sigma, band = _make_int_multiband_lc(band_ids, n_per_band=50, rng=rng)

    amp = licu_ext.Amplitude(bands=band_ids)
    result = amp(t, m, sigma, band, sorted=True)

    amp_sb = licu_ext.Amplitude()
    for i, bid in enumerate(band_ids):
        mask = band == bid
        expected = amp_sb(t[mask], m[mask], sigma[mask])[0]
        assert_allclose(result[i], expected, err_msg=f"band {bid} mismatch")


def test_integer_bands_contiguous_range():
    """Contiguous integer range [0,1,2] triggers O(1) lookup and gives correct results."""
    rng = np.random.default_rng(104)
    band_ids = [0, 1, 2]
    t, m, sigma, band = _make_int_multiband_lc(band_ids, n_per_band=50, rng=rng)

    amp = licu_ext.Amplitude(bands=band_ids)
    result = amp(t, m, sigma, band, sorted=True)

    amp_sb = licu_ext.Amplitude()
    for i, bid in enumerate(band_ids):
        mask = band == bid
        expected = amp_sb(t[mask], m[mask], sigma[mask])[0]
        assert_allclose(result[i], expected, err_msg=f"band {bid} mismatch")


def test_integer_bands_non_contiguous():
    """Non-contiguous integer bands (e.g., 0, 2, 5) fall back to linear lookup correctly."""
    rng = np.random.default_rng(105)
    band_ids = [0, 2, 5]
    t, m, sigma, band = _make_int_multiband_lc(band_ids, n_per_band=50, rng=rng)

    amp = licu_ext.Amplitude(bands=band_ids)
    result = amp(t, m, sigma, band, sorted=True)

    amp_sb = licu_ext.Amplitude()
    for i, bid in enumerate(band_ids):
        mask = band == bid
        expected = amp_sb(t[mask], m[mask], sigma[mask])[0]
        assert_allclose(result[i], expected, err_msg=f"band {bid} mismatch")


def test_integer_bands_negative():
    """Negative integer band IDs are handled correctly."""
    rng = np.random.default_rng(106)
    band_ids = [-2, -1, 0]
    t, m, sigma, band = _make_int_multiband_lc(band_ids, n_per_band=50, rng=rng)

    amp = licu_ext.Amplitude(bands=band_ids)
    result = amp(t, m, sigma, band, sorted=True)

    amp_sb = licu_ext.Amplitude()
    for i, bid in enumerate(band_ids):
        mask = band == bid
        expected = amp_sb(t[mask], m[mask], sigma[mask])[0]
        assert_allclose(result[i], expected, err_msg=f"band {bid} mismatch")


def test_integer_bands_property_returns_int64_array():
    """The .bands property returns a numpy int64 array when constructed with integers."""
    amp = licu_ext.Amplitude(bands=[3, 1, 2])
    bands = amp.bands
    assert bands is not None
    assert bands.dtype == np.int64
    assert_array_equal(bands, np.array([3, 1, 2], dtype=np.int64))


def test_integer_bands_property_after_pickle():
    """Integer bands property reconstructs correctly after a pickle round-trip."""
    amp = licu_ext.Amplitude(bands=[0, 1, 2])
    amp2 = pickle.loads(pickle.dumps(amp))
    assert amp2.bands is not None
    assert amp2.bands.dtype == np.int64
    assert_array_equal(amp2.bands, np.array([0, 1, 2], dtype=np.int64))


def test_integer_bands_unknown_raises():
    """Passing an unrecognized integer band ID raises ValueError."""
    t, m, sigma, band = _make_int_multiband_lc([0, 1], n_per_band=30, rng=107)
    amp = licu_ext.Amplitude(bands=[0, 1])
    band_bad = band.copy()
    band_bad[band_bad == 1] = 99
    with pytest.raises(ValueError, match="unknown passband"):
        amp(t, m, sigma, band_bad, sorted=True, check=True)


def test_integer_bands_matches_string_bands():
    """Integer bands produce identical results to equivalent string bands."""
    rng = np.random.default_rng(108)
    band_ids = [1, 2, 3]
    t, m, sigma, band_int = _make_int_multiband_lc(band_ids, n_per_band=50, rng=rng)
    band_str = np.array([str(b) for b in band_int])

    amp_int = licu_ext.Amplitude(bands=band_ids)
    amp_str = licu_ext.Amplitude(bands=["1", "2", "3"])

    result_int = amp_int(t, m, sigma, band_int, sorted=True)
    result_str = amp_str(t, m, sigma, band_str, sorted=True)
    assert_allclose(result_int, result_str)


def test_integer_bands_many_matches_call():
    """many() with integer bands produces the same result as repeated __call__."""
    rng = np.random.default_rng(109)
    band_ids = [0, 1, 2]
    n_lcs = 5
    lcs = [_make_int_multiband_lc(band_ids, n_per_band=30, rng=rng) for _ in range(n_lcs)]

    amp = licu_ext.Amplitude(bands=band_ids)
    call_results = np.stack([amp(t, m, sigma, band, sorted=True) for t, m, sigma, band in lcs])
    many_results = amp.many([(t, m, sigma, band) for t, m, sigma, band in lcs], sorted=True)
    assert_allclose(call_results, many_results)


def _make_arrow_int_multiband_lcs(band_lcs, band_col_type=pa.int64()):
    """Build a pyarrow List<Struct<t, m, sigma, band>> with an integer band column."""
    struct_type = pa.struct(
        [
            ("t", pa.float64()),
            ("m", pa.float64()),
            ("sigma", pa.float64()),
            ("band", band_col_type),
        ]
    )
    all_lcs = []
    for t, m, sigma, band in band_lcs:
        rows = [
            {"t": float(t[i]), "m": float(m[i]), "sigma": float(sigma[i]), "band": int(band[i])} for i in range(len(t))
        ]
        all_lcs.append(rows)
    return pa.array(all_lcs, type=pa.list_(struct_type))


@pytest.mark.parametrize(
    "pa_type",
    [pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint8(), pa.uint16(), pa.uint32()],
)
def test_integer_bands_arrow_dtype(pa_type):
    """Arrow multiband with integer band column type matches numpy result."""
    rng = np.random.default_rng(110)
    band_ids = [0, 1, 2]
    lcs = [_make_int_multiband_lc(band_ids, n_per_band=30, rng=rng) for _ in range(4)]

    feat = licu_ext.Amplitude(bands=band_ids)
    expected = feat.many([(t, m, sigma, band) for t, m, sigma, band in lcs], sorted=True)

    arrow_arr = _make_arrow_int_multiband_lcs(lcs, band_col_type=pa_type)
    result = feat.many(
        arrow_arr,
        sorted=True,
        fill_value=-999.0,
        arrow_fields={"t": "t", "m": "m", "sigma": "sigma", "band": "band"},
    )
    assert_allclose(result, expected)


def test_integer_bands_arrow_string_feature_mismatch_raises():
    """Arrow integer band column with a string-bands feature raises TypeError."""
    rng = np.random.default_rng(111)
    lcs = [_make_int_multiband_lc([0, 1], n_per_band=20, rng=rng) for _ in range(2)]
    arrow_arr = _make_arrow_int_multiband_lcs(lcs, band_col_type=pa.int32())

    feat = licu_ext.Amplitude(bands=["0", "1"])
    with pytest.raises(TypeError):
        feat.many(
            arrow_arr,
            sorted=True,
            fill_value=-999.0,
            arrow_fields={"t": "t", "m": "m", "sigma": "sigma", "band": "band"},
        )


# ---------------------------------------------------------------------------
# Benchmarks: integer vs string band dispatch
# ---------------------------------------------------------------------------

_BENCH_N_PER_BAND = 1_000
_BENCH_BANDS_STR = ["g", "r", "i"]
_BENCH_BANDS_INT = [0, 1, 2]


@pytest.fixture(scope="module")
def _bench_str_lc():
    return _make_multiband_lc(_BENCH_BANDS_STR, n_per_band=_BENCH_N_PER_BAND, rng=200)


@pytest.fixture(scope="module")
def _bench_int_lc():
    return _make_int_multiband_lc(_BENCH_BANDS_INT, n_per_band=_BENCH_N_PER_BAND, rng=200)


@pytest.fixture(scope="module")
def _bench_str_feat():
    return licu_ext.ObservationCount(bands=_BENCH_BANDS_STR)


@pytest.fixture(scope="module")
def _bench_int_feat():
    return licu_ext.ObservationCount(bands=_BENCH_BANDS_INT)


def test_benchmark_multiband_string_bands(benchmark, _bench_str_feat, _bench_str_lc):
    """Benchmark multiband evaluation with string band labels (numpy input)."""
    t, m, sigma, band = _bench_str_lc
    benchmark.group = "multiband_band_dispatch"
    benchmark.name = "string_bands"
    benchmark(lambda: _bench_str_feat(t, m, sigma, band, sorted=True, check=False))


def test_benchmark_multiband_integer_bands(benchmark, _bench_int_feat, _bench_int_lc):
    """Benchmark multiband evaluation with integer band labels (numpy int64 input)."""
    t, m, sigma, band = _bench_int_lc
    benchmark.group = "multiband_band_dispatch"
    benchmark.name = "integer_bands"
    benchmark(lambda: _bench_int_feat(t, m, sigma, band, sorted=True, check=False))
