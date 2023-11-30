import sys

import numpy as np
import pytest

from light_curve.light_curve_py import RainbowRisingFit


def bb_nu(wave_aa, T):
    nu = 3e10 / (wave_aa * 1e-8)
    return 2 * 6.626e-27 * nu**3 / 3e10**2 / np.expm1(6.626e-27 * nu / (1.38e-16 * T))


@pytest.mark.skipif(sys.version_info < (3, 8), reason="iminuit requires Python >= 3.8")

# Here parameters are degenerate, we cannot compare true values to the computed ones
def test_noisy_no_baseline():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 5.0
    Tmin = 5e3
    delta_T = 10e3
    k_sig = 4.0

    t = np.sort(rng.uniform(reference_time - 3 * rise_time, reference_time, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))
    waves = np.array([band_wave_aa[b] for b in band])

    temp = Tmin + delta_T / (1.0 + np.exp((t - reference_time) / k_sig))
    lum = amplitude / (1.0 + np.exp(-(t - reference_time) / rise_time))

    flux = np.pi * bb_nu(waves, temp) / (5.67e-5 * temp**4) * lum
    # S/N = 5 for minimum flux, scale for Poisson noise
    flux_err = np.sqrt(flux * np.min(flux) / 5.0)
    flux += rng.normal(0.0, flux_err)

    feature = RainbowRisingFit.from_angstrom(band_wave_aa, with_baseline=False)
    actual = feature(t, flux, sigma=flux_err, band=band)[-1]
    
    np.testing.assert_allclose(actual, 1, rtol=0.05)


@pytest.mark.skipif(sys.version_info < (3, 8), reason="iminuit requires Python >= 3.8")
def test_noisy_with_baseline():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}
    average_nu = np.mean([3e10 / (w * 1e-8) for w in band_wave_aa.values()])

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 10.0
    Tmin = 5e3
    delta_T = 10e3
    k_sig = 4.0
    baselines = {b: rng.exponential(scale=3 * amplitude / average_nu) for b in band_wave_aa}

    t = np.sort(rng.uniform(reference_time - 3 * rise_time, reference_time, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))
    waves = np.array([band_wave_aa[b] for b in band])

    temp = Tmin + delta_T / (1.0 + np.exp((t - reference_time) / k_sig))
    lum = amplitude / (1.0 + np.exp(-(t - reference_time) / rise_time))

    flux = np.pi * bb_nu(waves, temp) / (5.67e-5 * temp**4) * lum + np.array([baselines[b] for b in band])
    # S/N = 5 for minimum flux, scale for Poisson noise
    # We make noise a bit smaller because the optimization is not perfect
    flux_err = 0.1 * np.sqrt(flux * np.min(flux) / 5.0)
    flux += rng.normal(0.0, flux_err)

    feature = RainbowRisingFit.from_angstrom(band_wave_aa, with_baseline=True)

    actual = feature(t, flux, sigma=flux_err, band=band)[-1]

    np.testing.assert_allclose(actual, 1, rtol=0.05)
