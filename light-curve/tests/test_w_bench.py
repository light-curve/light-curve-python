import dataclasses
from functools import lru_cache, wraps
from itertools import count
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Union

import feets
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import curve_fit

import light_curve.light_curve_ext as lc_ext
import light_curve.light_curve_py as lc_py


@dataclasses.dataclass
class Data:
    name: str
    phot_type: str
    t: np.ndarray = dataclasses.field(repr=False)
    m: np.ndarray = dataclasses.field(repr=False)
    sigma: np.ndarray = dataclasses.field(repr=False)

    # Kinda static attribute
    phot_type_choices: frozenset = dataclasses.field(
        default=frozenset(["flux", "mag"]),
        init=False,
        repr=False,
    )

    # For '*' operator
    def __iter__(self):
        def gen():
            yield self.t
            yield self.m
            yield self.sigma

        return gen()

    def __post_init__(self):
        self.name = str(self.name)
        assert self.phot_type in self.phot_type_choices
        assert self.t.size == self.m.size == self.sigma.size
        assert self.t.dtype == self.m.dtype == self.sigma.dtype


def gen_data_from_test_data_path(
    paths: Iterator[Union[Path, str]], *, n: Optional[int] = None, convert_to_flux: bool = False
) -> Generator[Data, None, None]:
    if n is None:
        take_n = count()
    else:
        take_n = range(n)
    for _, csv_file in zip(take_n, paths):
        df = pd.read_csv(csv_file)

        # Drop repeated values
        idx = np.diff(df["time"], prepend=0) > 0
        df = df[idx]

        t = df["time"].to_numpy()
        if "mag" in df.columns:
            m = df["mag"].to_numpy()
            sigma = df["magerr"].to_numpy()
            if convert_to_flux:
                m = 10 ** (-0.4 * m)
                sigma = 0.4 * np.log(10.0) * sigma * m
                phot_type = "flux"
            else:
                phot_type = "mag"
        elif "flux" in df.columns:
            m = df["flux"].to_numpy()
            sigma = df["fluxerr"].to_numpy()
            phot_type = "flux"
        else:
            raise ValueError(f"No mag neither flux column in {csv_file}")
        yield Data(csv_file, phot_type, t, m, sigma)


@lru_cache(maxsize=1)
def get_test_data_from_issues() -> List[Data]:
    data_root = Path(__file__).parent / "light-curve-test-data/from-issues"
    return list(gen_data_from_test_data_path(data_root.glob("**/*.csv")))


@lru_cache()
def get_first_n_snia_data(*, n: Optional[int] = None, convert_to_flux: bool = False) -> List[Data]:
    data_root = Path(__file__).parent / "light-curve-test-data/SNIa"
    with open(data_root / "snIa_bandg_minobs10_beforepeak3_afterpeak4.csv") as fh:
        file_names = frozenset(fh.read().split())
    light_curve_dir = data_root / "light-curves"
    paths = (path for path in light_curve_dir.glob("*.csv") if path.stem in file_names)
    return list(gen_data_from_test_data_path(paths, n=n, convert_to_flux=convert_to_flux))


class _Test:
    # Feature name must be updated in child classes
    name = None
    # Argument tuple for the feature constructor
    args = ()

    py_feature = None

    # Specify method for a naive implementation
    naive = None

    # Specify `feets` feature name
    feets_feature = None
    feets_extractor = None
    # Specify a `str` with a reason why skip this test
    feets_skip_test = False

    # Which types of pre-generated light-curves to use
    phot_types = Data.phot_type_choices

    def setup_method(self):
        self.rust = getattr(lc_ext, self.name)(*self.args)
        try:
            self.py_feature = getattr(lc_py, self.name)(*self.args)
        except AttributeError:
            pass

        if self.feets_feature is not None:
            self.feets_extractor = feets.FeatureSpace(only=[self.feets_feature], data=["time", "magnitude", "error"])

    # Default values of `assert_allclose`
    rtol = np.finfo(np.float32).resolution  # 1e-6
    atol = 0

    # Default values for random light curve generation
    n_obs = 1000
    t_min = 0.0
    t_max = 1000.0
    m_min = 15.0
    m_max = 21.0
    sigma_min = 0.01
    sigma_max = 0.2

    add_to_all_features = True

    def random_data(self):
        t = np.sort(np.random.uniform(self.t_min, self.t_max, self.n_obs))
        m = np.random.uniform(self.m_min, self.m_max, self.n_obs)
        sigma = np.random.uniform(self.sigma_min, self.sigma_max, self.n_obs)
        return Data("random", "mag", t, m, sigma)

    def from_issues_data(self):
        for data in get_test_data_from_issues():
            if data.phot_type not in self.phot_types:
                continue
            yield data

    def snia_data(self, n: Optional[int] = None):
        for data in get_first_n_snia_data(n=n):
            if data.phot_type not in self.phot_types:
                continue
            yield data

    def data_gen(self) -> Generator[Data, None, None]:
        yield self.random_data()
        yield from self.from_issues_data()
        yield from self.snia_data(n=10)

    def test_feature_length(self, subtests):
        for data in self.data_gen():
            with subtests.test(lc_name=data):
                result = self.rust(*data, sorted=None)
                assert len(result) == len(self.rust.names) == len(self.rust.descriptions)

    def test_close_to_lc_py(self, subtests):
        if self.py_feature is None:
            pytest.skip("No matched light_curve_py class for the feature")
        for data in self.data_gen():
            with subtests.test(data=data):
                assert_allclose(self.rust(*data), self.py_feature(*data), rtol=self.rtol, atol=self.atol)

    def test_benchmark_rust(self, benchmark):
        t, m, sigma = self.random_data()
        benchmark.group = str(type(self).__name__)
        benchmark(self.rust, t, m, sigma, sorted=True, check=False)

    def test_benchmark_lc_py(self, benchmark):
        if self.py_feature is None:
            pytest.skip("No matched light_curve_py class for the feature")

        t, m, sigma = self.random_data()

        benchmark.group = str(type(self).__name__)
        benchmark(self.py_feature, t, m, sigma, sorted=True, check=False)

    def test_close_to_naive(self, subtests):
        if self.naive is None:
            pytest.skip("No naive implementation for the feature")
        for data in self.data_gen():
            with subtests.test(data=data):
                assert_allclose(self.rust(*data), self.naive(*data), rtol=self.rtol, atol=self.atol)

    def test_benchmark_naive(self, benchmark):
        if self.naive is None:
            pytest.skip("No naive implementation for the feature")

        t, m, sigma = self.random_data()

        benchmark.group = type(self).__name__
        benchmark(self.naive, t, m, sigma)

    def feets(self, t, m, sigma):
        _, result = self.feets_extractor.extract(t, m, sigma)
        return result

    def test_close_to_feets(self, subtests):
        if self.feets_extractor is None:
            pytest.skip("No feets feature provided")
        if self.feets_skip_test:
            pytest.skip("feets is expected to be different from light_curve, reason: " + self.feets_skip_test)

        for data in self.data_gen():
            with subtests.test(data=data):
                assert_allclose(self.rust(*data)[:1], self.feets(*data)[:1], rtol=self.rtol, atol=self.atol)

    def test_benchmark_feets(self, benchmark):
        if self.feets_extractor is None:
            pytest.skip("No feets feature provided")

        t, m, sigma = self.random_data()

        benchmark.group = type(self).__name__
        benchmark(self.feets, t, m, sigma)


class TestAmplitude(_Test):
    name = "Amplitude"

    def naive(self, t, m, sigma):
        return 0.5 * (np.max(m) - np.min(m))


class TestAndersonDarlingNormal(_Test):
    name = "AndersonDarlingNormal"

    feets_feature = "AndersonDarling"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        return stats.anderson(m).statistic * (1.0 + 4.0 / m.size - 25.0 / m.size**2)


if lc_ext._built_with_gsl:

    class TestBazinFit(_Test):
        name = "BazinFit"
        args = ("lmsder", None, 20)
        rtol = 1e-4  # Precision used in the feature implementation

        add_to_all_features = False  # in All* random data is used

        phot_types = frozenset(["flux"])

        @staticmethod
        def _model(t, a, b, t0, rise, fall):
            dt = t - t0
            return b + a * np.exp(-dt / fall) / (1.0 + np.exp(-dt / rise))

        def _params(self):
            a = 100
            c = 100
            t0 = 0.5 * (self.t_min + self.t_max)
            rise = 0.1 * (self.t_max - self.t_min)
            fall = 0.2 * (self.t_max - self.t_min)
            return a, c, t0, rise, fall

        # Random data yields to random results because target function has a lot of local minima
        # BTW, this test shouldn't use fixed random seed because the curve has good enough S/N to be fitted for any give
        # noise sample
        def random_data(self):
            rng = np.random.default_rng(0)
            t = np.linspace(self.t_min, self.t_max, self.n_obs)
            model = self._model(t, *self._params())
            sigma = np.sqrt(model)
            m = model + sigma * rng.normal(size=self.n_obs)
            return Data("Bazin+noise", "flux", t, m, sigma)

        # Keep random data only
        def data_gen(self) -> Generator[Data, None, None]:
            yield self.random_data()
            # All these fail =(
            # yield from get_first_n_snia_data(n=10, convert_to_flux=True)

        def naive(self, t, m, sigma):
            params, _cov = curve_fit(
                self._model,
                xdata=t,
                ydata=m,
                sigma=sigma,
                xtol=self.rtol,
                # We give really good parameters estimation!
                # p0=self._params(),
                # The same parameter estimations we use in Rust
                p0=(0.5 * np.ptp(m), np.min(m), t[np.argmax(m)], 0.5 * np.ptp(t), 0.5 * np.ptp(t)),
            )
            reduced_chi2 = np.sum(np.square((self._model(t, *params) - m) / sigma)) / (t.size - params.size)
            return_value = tuple(params) + (reduced_chi2,)
            return return_value


class TestBeyond1Std(_Test):
    nstd = 1.0

    name = "BeyondNStd"
    args = (nstd,)

    feets_feature = "Beyond1Std"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        mean = np.mean(m)
        interval = self.nstd * np.std(m, ddof=1)
        return np.count_nonzero(np.abs(m - mean) > interval) / m.size


class TestCusum(_Test):
    name = "Cusum"

    feets_feature = "Rcs"
    feets_skip_test = "feets uses biased statistics"


class TestEta(_Test):
    name = "Eta"

    def naive(self, t, m, sigma):
        return np.sum(np.square(m[1:] - m[:-1])) / (np.var(m, ddof=0) * m.size)


class TestEtaE(_Test):
    name = "EtaE"

    feets_feature = "Eta_e"
    feets_skip_test = "feets fixed EtaE from the original paper in different way"

    def naive(self, t, m, sigma):
        return (
            np.sum(np.square((m[1:] - m[:-1]) / (t[1:] - t[:-1])))
            * (t[-1] - t[0]) ** 2
            / (np.var(m, ddof=0) * m.size * (m.size - 1) ** 2)
        )


class TestExcessVariance(_Test):
    name = "ExcessVariance"

    def naive(self, t, m, sigma):
        return (np.var(m, ddof=1) - np.mean(sigma**2)) / np.mean(m) ** 2


class TestInterPercentileRange(_Test):
    quantile = 0.25

    name = "InterPercentileRange"
    args = (quantile,)

    feets_feature = "Q31"
    feets_skip_test = "feets uses different quantile type"


class TestOtsuSplit(_Test):
    name = "OtsuSplit"


def magnitude_function(func, *args, **kwargs):
    @wraps(func)
    def decorated(t, m, sigma=None, sorted=None, check=None):
        return func(m)

    return decorated


class TestOtsuSplitThreshold(_Test):
    def setup_method(self):
        self.py_feature = magnitude_function(lc_py.OtsuSplit.threshold)
        self.rust = magnitude_function(lc_ext.OtsuSplit.threshold)

    def test_feature_length(self):
        """Not a real feature extractor, no need to test length"""
        pass


class TestKurtosis(_Test):
    name = "Kurtosis"

    feets_feature = "SmallKurtosis"
    feets_skip_test = "feets uses equation for unbiased kurtosis, but put biased standard deviation there"

    def naive(self, t, m, sigma):
        return stats.kurtosis(m, fisher=True, bias=False)


class TestLinearTrend(_Test):
    name = "LinearTrend"

    feets_feature = "LinearTrend"

    def naive(self, t, m, sigma):
        (slope, _), ((slope_sigma2, _), _) = np.polyfit(t, m, deg=1, cov=True)
        sigma_noise = np.sqrt(np.polyfit(t, m, deg=1, full=True)[1][0] / (t.size - 2))
        return np.array([slope, np.sqrt(slope_sigma2), sigma_noise])


def generate_test_magnitude_percentile_ratio(quantile_numerator, quantile_denumerator, feets_feature):
    return type(
        f"TestMagnitudePercentageRatio{int(quantile_numerator * 100):d}",
        (_Test,),
        dict(
            args=(quantile_numerator, quantile_denumerator),
            quantile_numerator=quantile_numerator,
            quantile_denumerator=quantile_denumerator,
            name="MagnitudePercentageRatio",
            feets_feature=feets_feature,
            feets_skip_test="feets uses different quantile type",
        ),
    )


FluxPercentileRatioMid20 = generate_test_magnitude_percentile_ratio(0.40, 0.05, "FluxPercentileRatioMid20")
FluxPercentileRatioMid50 = generate_test_magnitude_percentile_ratio(0.25, 0.05, "FluxPercentileRatioMid50")
FluxPercentileRatioMid80 = generate_test_magnitude_percentile_ratio(0.10, 0.05, "FluxPercentileRatioMid80")


class TestMaximumSlope(_Test):
    name = "MaximumSlope"

    feets_feature = "MaxSlope"

    def naive(self, t, m, sigma):
        return np.max(np.abs((m[1:] - m[:-1]) / (t[1:] - t[:-1])))


class TestMean(_Test):
    name = "Mean"

    feets_feature = "Mean"

    def naive(self, t, m, sigma):
        return np.mean(m)


class TestMeanVariance(_Test):
    name = "MeanVariance"

    feets_feature = "Meanvariance"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        return np.std(m, ddof=1) / np.mean(m)


class TestMedian(_Test):
    name = "Median"

    def naive(self, t, m, sigma):
        return np.median(m)


class TestMedianAbsoluteDeviation(_Test):
    name = "MedianAbsoluteDeviation"

    feets_feature = "MedianAbsDev"


class TestMedianBufferRangePercentage(_Test):
    # feets says it uses 0.1 of amplitude (a half range between max and min),
    # but factually it uses 0.1 of full range between max and min
    quantile = 0.2

    name = "MedianBufferRangePercentage"
    args = (quantile,)

    feets_feature = "MedianBRP"


class TestPercentAmplitude(_Test):
    name = "PercentAmplitude"

    feets_feature = "PercentAmplitude"
    feets_skip_test = "feets divides value by median"

    def naive(self, t, m, sigma):
        median = np.median(m)
        return max(np.max(m) - median, median - np.min(m))


class TestPercentDifferenceMagnitudePercentile(_Test):
    quantile = 0.05

    name = "PercentDifferenceMagnitudePercentile"
    args = (quantile,)

    feets_feature = "PercentDifferenceFluxPercentile"
    feets_skip_test = "feets uses different quantile type"


class TestReducedChi2(_Test):
    name = "ReducedChi2"

    def naive(self, t, m, sigma):
        w = 1.0 / np.square(sigma)
        return np.sum(np.square(m - np.average(m, weights=w)) * w) / (m.size - 1)


class TestSkew(_Test):
    name = "Skew"

    feets_feature = "Skew"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        return stats.skew(m, bias=False)


class TestStandardDeviation(_Test):
    name = "StandardDeviation"

    feets_feature = "Std"
    feets_skip_test = "feets uses biased statistics"

    def naive(self, t, m, sigma):
        return np.std(m, ddof=1)


class TestStetsonK(_Test):
    name = "StetsonK"

    feets_feature = "StetsonK"

    def naive(self, t, m, sigma):
        x = (m - np.average(m, weights=1.0 / sigma**2)) / sigma
        return np.sum(np.abs(x)) / np.sqrt(np.sum(np.square(x)) * m.size)


class TestWeightedMean(_Test):
    name = "WeightedMean"

    def naive(self, t, m, sigma):
        return np.average(m, weights=1.0 / sigma**2)


class TestAllPy(_Test):
    # Most of the features are for mags
    phot_types = frozenset(["mag"])

    def setup_method(self):
        features = []
        py_features = []
        for cls in _Test.__subclasses__():
            if cls.name is None:
                continue

            try:
                py_features.append(getattr(lc_py, cls.name)(*cls.args))
            except AttributeError:
                continue
            features.append(getattr(lc_ext, cls.name)(*cls.args))
        self.rust = lc_ext.Extractor(*features)
        self.py_feature = lc_py.Extractor(*py_features)


class TestAllNaive(_Test):
    # Most of the features are for mags
    phot_types = frozenset(["mag"])

    def setup_method(self):
        features = []
        self.naive_features = []
        for cls in _Test.__subclasses__():
            if cls.naive is None or cls.name is None:
                continue
            if not cls.add_to_all_features:
                continue
            if not cls.add_to_all_features:
                continue
            features.append(getattr(lc_ext, cls.name)(*cls.args))
            self.naive_features.append(cls().naive)
        self.rust = lc_ext.Extractor(*features)

    def naive(self, t, m, sigma):
        return np.concatenate([np.atleast_1d(f(t, m, sigma)) for f in self.naive_features])


class TestAllFeets(_Test):
    feets_skip_test = "skip for TestAllFeets"

    # Most of the features are for mags
    phot_types = frozenset(["mag"])

    def setup_method(self):
        features = []
        feets_features = []
        for cls in _Test.__subclasses__():
            if cls.feets_feature is None or cls.name is None:
                continue
            if not cls.add_to_all_features:
                continue
            if not cls.add_to_all_features:
                continue
            features.append(getattr(lc_ext, cls.name)(*cls.args))
            feets_features.append(cls.feets_feature)
        self.rust = lc_ext.Extractor(*features)
        self.feets_extractor = feets.FeatureSpace(only=feets_features, data=["time", "magnitude", "error"])
