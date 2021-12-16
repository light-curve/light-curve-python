from itertools import product

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result


import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from light_curve.light_curve_ext import DmDt


def random_lc(n, sigma=True, rng=None, dtype=np.float64):
    rng = np.random.default_rng(rng)
    t = np.sort(np.asarray(rng.uniform(0, 10, n), dtype=dtype))
    m = np.asarray(rng.normal(0, 1, n), dtype=dtype)
    lc = (t, m)
    if sigma:
        sigma = np.asarray(rng.uniform(0.01, 0.1, n), dtype=dtype)
        lc += (sigma,)
    return lc


def sine_lc(n, sigma=True, dtype=np.float64):
    t = np.asarray(np.linspace(0, 10, n), dtype=dtype)
    m = np.sin(t)
    lc = (t, m)
    if sigma:
        sigma = np.full_like(t, 0.1)
        lc += (sigma,)
    return lc


def sliced(a, step=2):
    """Mix with random data and slice to original data"""
    assert step > 0
    n = a.size
    rng = np.random.default_rng()
    random_data = np.asarray(rng.normal(0, 1, (step - 1, n)), dtype=a.dtype)
    mixed = np.vstack([a[::-1], random_data]).T.reshape(-1).copy()
    s = mixed[-step::-step]
    assert not s.flags.owndata
    assert_array_equal(s, a)
    return s


def test_dmdt_count_dt_three_obs():
    dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=np.log10(3), max_abs_dm=1, lgdt_size=2, dm_size=32, norm=[])

    t = np.array([0, 1, 2], dtype=np.float32)

    desired = np.array([2, 1])
    actual = dmdt.count_dt(t)

    assert_array_equal(actual, desired)


def test_log_linear_grids():
    lc = random_lc(101)

    min_lgdt = -1
    max_lgdt = 1
    lgdt_size = 32
    max_abs_dm = 2
    dm_size = 32

    min_dt = 10 ** min_lgdt
    max_dt = 10 ** max_lgdt

    dt_grid = np.logspace(min_lgdt, max_lgdt, lgdt_size + 1)
    dm_grid = np.linspace(-max_abs_dm, max_abs_dm, dm_size + 1)

    dmdt_from_borders = DmDt.from_borders(
        min_lgdt=min_lgdt, max_lgdt=max_lgdt, max_abs_dm=max_abs_dm, lgdt_size=lgdt_size, dm_size=dm_size
    )
    dmdt_auto = DmDt(dt=dt_grid, dm=dm_grid, dt_type="auto", dm_type="auto")
    dmdt_log_linear = DmDt(dt=dt_grid, dm=dm_grid, dt_type="log", dm_type="linear")
    dmdt_asis = DmDt(dt=dt_grid, dm=dm_grid, dt_type="asis", dm_type="asis")

    for dmdt in (
        dmdt_from_borders,
        dmdt_auto,
        dmdt_log_linear,
        dmdt_asis,
    ):
        assert_allclose(dmdt.min_dt, min_dt)
        assert_allclose(dmdt.max_dt, max_dt)
        assert_allclose(dmdt.min_dm, -max_abs_dm)
        assert_allclose(dmdt.max_dm, max_abs_dm)
        assert_allclose(dmdt.dt_grid, dt_grid)
        assert_allclose(dmdt.dm_grid, dm_grid)

    points = dmdt_from_borders.points(lc[0], lc[1])
    gausses = dmdt_from_borders.gausses(*lc)
    for dmdt in (
        dmdt_auto,
        dmdt_log_linear,
        dmdt_asis,
    ):
        assert_allclose(dmdt.points(lc[0], lc[1]), points)
        assert_allclose(dmdt.gausses(*lc), gausses)


@pytest.mark.parametrize("lc", [sine_lc(101), sine_lc(101, dtype=np.float32)])
def test_dmdt_count_dt_contiguous_non(lc):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[])
    desired = dmdt.count_dt(lc[0])
    actual = dmdt.count_dt(sliced(lc[0]))
    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101), random_lc(101), sine_lc(101, dtype=np.float32)])
def test_dmdt_count_dt_many_one(lc):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[])

    desired = dmdt.count_dt(lc[0])
    assert np.any(desired != 0)
    actual = dmdt.count_dt_many([lc[0]])

    assert actual.shape[0] == 1
    assert_array_equal(actual[0], desired)


@pytest.mark.parametrize(
    "lcs",
    [
        [sine_lc(101), sine_lc(11)],
        [random_lc(101), random_lc(101), random_lc(11)],
        [sine_lc(101, dtype=np.float32), sine_lc(11, dtype=np.float32)],
    ],
)
def test_dmdt_count_dt_many(lcs):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[])

    desired = [dmdt.count_dt(t) for t, *_ in lcs]
    actual = dmdt.count_dt_many([t for t, *_ in lcs])

    assert_array_equal(actual, desired)


def test_dmdt_points_three_obs():
    dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=np.log10(3), max_abs_dm=3, lgdt_size=2, dm_size=4, norm=[])

    t = np.array([0, 1, 2], dtype=np.float32)
    m = np.array([0, 1, 2], dtype=np.float32)

    desired = np.array(
        [
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ]
    )
    actual = dmdt.points(t, m)

    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101, False), sine_lc(101, False, dtype=np.float32)])
def test_dmdt_points_contiguous_non(lc):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[])
    desired = dmdt.points(*lc)
    t, m = lc
    t = sliced(t)
    m = sliced(m)
    actual = dmdt.points(t, m)
    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101, False), random_lc(101, False), sine_lc(101, False, dtype=np.float32)])
@pytest.mark.parametrize("norm", [[], ["dt"], ["max"], ["dt", "max"]])
def test_dmdt_points_many_one(lc, norm):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=norm)

    desired = dmdt.points(*lc)
    assert np.any(desired != 0)
    actual = dmdt.points_many([lc])

    assert actual.shape[0] == 1
    assert_array_equal(actual[0], desired)


@pytest.mark.parametrize(
    "lcs",
    [
        [sine_lc(101, False), sine_lc(11, False)],
        [random_lc(101, False), random_lc(101, False), random_lc(11, False)],
        [sine_lc(101, False, dtype=np.float32), sine_lc(11, False, dtype=np.float32)],
    ],
)
@pytest.mark.parametrize("norm", [[], ["dt"], ["max"], ["dt", "max"]])
def test_dmdt_points_many(lcs, norm):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=norm)

    desired = [dmdt.points(*lc) for lc in lcs]
    actual = dmdt.points_many(lcs)

    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101), sine_lc(101, dtype=np.float32)])
def test_dmdt_gausses_contiguous_non(lc):
    dmdt = DmDt.from_borders(min_lgdt=-1, max_lgdt=1, max_abs_dm=2, lgdt_size=32, dm_size=32, norm=[])
    desired = dmdt.gausses(*lc)
    t, m, sigma = lc
    t = sliced(t)
    m = sliced(m)
    sigma = sliced(sigma)
    actual = dmdt.gausses(t, m, sigma)
    assert_array_equal(actual, desired)


@pytest.mark.parametrize("lc", [sine_lc(101), random_lc(101), sine_lc(101, dtype=np.float32)])
@pytest.mark.parametrize("norm", [[], ["dt"], ["max"], ["dt", "max"]])
@pytest.mark.parametrize("approx_erf", [True, False])
def test_dmdt_gausses_many_one(lc, norm, approx_erf):
    dmdt = DmDt.from_borders(
        min_lgdt=-1,
        max_lgdt=1,
        max_abs_dm=2,
        lgdt_size=32,
        dm_size=32,
        norm=norm,
        approx_erf=approx_erf,
    )

    desired = dmdt.gausses(*lc)
    assert np.any(desired != 0)
    actual = dmdt.gausses_many([lc])

    assert actual.shape[0] == 1
    assert_array_equal(actual[0], desired)


@pytest.mark.parametrize(
    "lcs",
    [
        [sine_lc(101), sine_lc(11)],
        [random_lc(101), random_lc(101), random_lc(11)],
        [sine_lc(101, dtype=np.float32), sine_lc(11, dtype=np.float32)],
    ],
)
@pytest.mark.parametrize("norm", [[], ["dt"], ["max"], ["dt", "max"]])
@pytest.mark.parametrize("approx_erf", [True, False])
def test_dmdt_gausses_many(lcs, norm, approx_erf):
    dmdt = DmDt.from_borders(
        min_lgdt=-1,
        max_lgdt=1,
        max_abs_dm=2,
        lgdt_size=32,
        dm_size=32,
        norm=norm,
        approx_erf=approx_erf,
    )

    desired = [dmdt.gausses(*lc) for lc in lcs]
    actual = dmdt.gausses_many(lcs)

    assert_array_equal(actual, desired)


@pytest.mark.parametrize("t_dtype,m_dtype", product(*[[np.float32, np.float64]] * 2))
def test_dmdt_points_dtype(t_dtype, m_dtype):
    t = np.linspace(0, 1, 11, dtype=t_dtype)
    m = np.asarray(t, dtype=m_dtype)
    dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=1, max_abs_dm=1, lgdt_size=2, dm_size=2, norm=[])
    if t_dtype is m_dtype:
        context = nullcontext()
    else:
        context = pytest.raises(TypeError)
    with context:
        dmdt.points(t, m)


@pytest.mark.parametrize("t_dtype,m_dtype,sigma_dtype", product(*[[np.float32, np.float64]] * 3))
def test_dmdt_gausses_dtype(t_dtype, m_dtype, sigma_dtype):
    t = np.linspace(1, 2, 11, dtype=t_dtype)
    m = np.asarray(t, dtype=m_dtype)
    sigma = np.asarray(t, dtype=sigma_dtype)
    dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=1, max_abs_dm=1, lgdt_size=2, dm_size=2, norm=[])
    if t_dtype is m_dtype is sigma_dtype:
        context = nullcontext()
    else:
        context = pytest.raises(TypeError)
    with context:
        dmdt.gausses(t, m, sigma)


@pytest.mark.parametrize("t1_dtype,m1_dtype,t2_dtype,m2_dtype", product(*[[np.float32, np.float64]] * 4))
def test_dmdt_points_many_dtype(t1_dtype, m1_dtype, t2_dtype, m2_dtype):
    t1 = np.linspace(1, 2, 11, dtype=t1_dtype)
    m1 = np.asarray(t1, dtype=m1_dtype)
    t2 = np.asarray(t1, dtype=t2_dtype)
    m2 = np.asarray(t1, dtype=m2_dtype)
    lcs = [(t1, m1), (t2, m2)]
    dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=1, max_abs_dm=1, lgdt_size=2, dm_size=2, norm=[])
    if t1_dtype is m1_dtype is t2_dtype is m2_dtype:
        context = nullcontext()
    else:
        context = pytest.raises(TypeError)
    with context:
        dmdt.points_many(lcs)
        dmdt.points_batches(lcs)


@pytest.mark.parametrize(
    "t1_dtype,m1_dtype,sigma1_dtype,t2_dtype,m2_dtype,sigma2_dtype", product(*[[np.float32, np.float64]] * 6)
)
def test_dmdt_gausses_many_dtype(t1_dtype, m1_dtype, sigma1_dtype, t2_dtype, m2_dtype, sigma2_dtype):
    t1 = np.linspace(1, 2, 11, dtype=t1_dtype)
    m1 = np.asarray(t1, dtype=m1_dtype)
    sigma1 = np.asarray(t1, dtype=sigma1_dtype)
    t2 = np.asarray(t1, dtype=t2_dtype)
    m2 = np.asarray(t1, dtype=m2_dtype)
    sigma2 = np.asarray(t1, dtype=sigma2_dtype)
    lcs = [(t1, m1, sigma1), (t2, m2, sigma2)]
    dmdt = DmDt.from_borders(min_lgdt=0, max_lgdt=1, max_abs_dm=1, lgdt_size=2, dm_size=2, norm=[])
    if t1_dtype is m1_dtype is sigma1_dtype is t2_dtype is m2_dtype is sigma2_dtype:
        context = nullcontext()
    else:
        context = pytest.raises(TypeError)
    with context:
        dmdt.gausses_many(lcs)
        dmdt.gausses_batches(lcs)
