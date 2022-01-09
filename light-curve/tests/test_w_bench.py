import feets
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats
from scipy.optimize import curve_fit

import light_curve.light_curve_ext as lc_ext
import light_curve.light_curve_py as lc_py


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

    def setup_method(self):
        self.rust = getattr(lc_ext, self.name)(*self.args)
        try:
            self.py_feature = getattr(lc_py, self.name)(*self.args)
        except AttributeError:
            pass

        if self.feets_feature is not None:
            self.feets_extractor = feets.FeatureSpace(only=[self.feets_feature], data=["time", "magnitude", "error"])

    # Default values of `assert_allclose`
    rtol = 1e-7
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

    def generate_data(self):
        t = np.sort(np.random.uniform(self.t_min, self.t_max, self.n_obs))
        m = np.random.uniform(self.m_min, self.m_max, self.n_obs)
        sigma = np.random.uniform(self.sigma_min, self.sigma_max, self.n_obs)
        return t, m, sigma

    def test_feature_length(self):
        t, m, sigma = self.generate_data()
        result = self.rust(t, m, sigma, sorted=None)
        assert len(result) == len(self.rust.names) == len(self.rust.descriptions)

    def test_close_to_lc_py(self):
        if self.py_feature is None:
            pytest.skip("No matched light_curve_py class for the feature")
        t, m, sigma = self.generate_data()
        assert_allclose(self.rust(t, m, sigma), self.py_feature(t, m, sigma), rtol=self.rtol, atol=self.atol)

    def test_benchmark_rust(self, benchmark):
        t, m, sigma = self.generate_data()

        benchmark.group = str(type(self).__name__)
        benchmark(self.rust, t, m, sigma, sorted=True, check=False)

    def test_benchmark_lc_py(self, benchmark):
        if self.py_feature is None:
            pytest.skip("No matched light_curve_py class for the feature")

        t, m, sigma = self.generate_data()

        benchmark.group = str(type(self).__name__)
        benchmark(self.py_feature, t, m, sigma, sorted=True, check=False)

    def test_close_to_naive(self):
        if self.naive is None:
            pytest.skip("No naive implementation for the feature")

        t, m, sigma = self.generate_data()
        assert_allclose(self.rust(t, m, sigma), self.naive(t, m, sigma), rtol=self.rtol, atol=self.atol)

    def test_benchmark_naive(self, benchmark):
        if self.naive is None:
            pytest.skip("No naive implementation for the feature")

        t, m, sigma = self.generate_data()

        benchmark.group = type(self).__name__
        benchmark(self.naive, t, m, sigma)

    def feets(self, t, m, sigma):
        _, result = self.feets_extractor.extract(t, m, sigma)
        return result

    def test_close_to_feets(self):
        if self.feets_extractor is None:
            pytest.skip("No feets feature provided")
        if self.feets_skip_test:
            pytest.skip("feets is expected to be different from light_curve, reason: " + self.feets_skip_test)

        t, m, sigma = self.generate_data()
        assert_allclose(self.rust(t, m, sigma)[:1], self.feets(t, m, sigma)[:1], rtol=self.rtol, atol=self.atol)

    def test_benchmark_feets(self, benchmark):
        if self.feets_extractor is None:
            pytest.skip("No feets feature provided")

        t, m, sigma = self.generate_data()

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
        return stats.anderson(m).statistic * (1.0 + 4.0 / m.size - 25.0 / m.size ** 2)


if lc_ext._built_with_gsl:

    class TestBazinFit(_Test):
        name = "BazinFit"
        args = ("lmsder", None, 20)
        rtol = 1e-4  # Precision used in the feature implementation

        add_to_all_features = False  # in All* random data is used

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
        def generate_data(self):
            rng = np.random.default_rng(0)
            t = np.linspace(self.t_min, self.t_max, self.n_obs)
            model = self._model(t, *self._params())
            sigma = np.sqrt(model)
            m = model + sigma * rng.normal(size=self.n_obs)
            return t, m, sigma

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
        return (np.var(m, ddof=1) - np.mean(sigma ** 2)) / np.mean(m) ** 2


class TestInterPercentileRange(_Test):
    quantile = 0.25

    name = "InterPercentileRange"
    args = (quantile,)

    feets_feature = "Q31"
    feets_skip_test = "feets uses different quantile type"


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
            quantile_numerator=quantile_numerator,
            quantile_denumerator=quantile_denumerator,
            name="MagnitudePercentageRatio",
            feets_skip_test="feets uses different quantile type",
        ),
    )


generate_test_magnitude_percentile_ratio(0.40, 0.05, "FluxPercentileRatioMid20")
generate_test_magnitude_percentile_ratio(0.25, 0.05, "FluxPercentileRatioMid50")
generate_test_magnitude_percentile_ratio(0.10, 0.05, "FluxPercentileRatioMid80")


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
        x = (m - np.average(m, weights=1.0 / sigma ** 2)) / sigma
        return np.sum(np.abs(x)) / np.sqrt(np.sum(np.square(x)) * m.size)


class TestWeightedMean(_Test):
    name = "WeightedMean"

    def naive(self, t, m, sigma):
        return np.average(m, weights=1.0 / sigma ** 2)


class TestAllPy(_Test):
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
