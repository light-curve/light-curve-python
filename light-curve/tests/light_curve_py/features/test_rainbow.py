import numpy as np

from light_curve.light_curve_py import RainbowFit


def test_noisy_no_baseline():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 5.0
    fall_time = 30.0
    Tmin = 5e3
    Tmax = 15e3
    k_sig = 4.0

    expected = [reference_time, amplitude, rise_time, fall_time, Tmin, Tmax, k_sig, 1.0]

    feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=False, temperature="sigmoid", bolometric="bazin")

    t = np.sort(rng.uniform(reference_time - 3 * rise_time, reference_time + 3 * fall_time, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))

    flux = feature.model(t, band, *expected)
    # S/N = 10 for minimum flux, scale for Poisson noise
    flux_err = np.sqrt(flux * np.min(flux)) / 10.0
    flux += rng.normal(0.0, flux_err)

    actual = feature(t, flux, sigma=flux_err, band=band)

    # import matplotlib.pyplot as plt
    # plt.scatter(t, flux, s=5, label="data")
    # plt.errorbar(t, flux, yerr=flux_err, ls="none", capsize=1)
    # plt.plot(t, feature.model(t, band, *expected), "x", label="expected")
    # plt.plot(t, feature.model(t, band, *actual), "*", label="actual")
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=0.1)


def test_noisy_with_baseline():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 5.0
    fall_time = 30.0
    Tmin = 5e3
    Tmax = 15e3
    k_sig = 4.0
    baselines = {b: 0.3 * amplitude + rng.exponential(scale=0.3 * amplitude) for b in band_wave_aa}

    expected = [reference_time, amplitude, rise_time, fall_time, Tmin, Tmax, k_sig, *baselines.values(), 1.0]

    feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=True, temperature="sigmoid", bolometric="bazin")

    t = np.sort(rng.uniform(reference_time - 3 * rise_time, reference_time + 3 * fall_time, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))

    flux = feature.model(t, band, *expected)
    # S/N = 10 for minimum flux, scale for Poisson noise
    flux_err = np.sqrt(flux * np.min(flux)) / 10.0
    flux += rng.normal(0.0, flux_err)

    actual = feature(t, flux, sigma=flux_err, band=band)

    # import matplotlib.pyplot as plt
    # plt.scatter(t, flux, s=5, label="data")
    # plt.errorbar(t, flux, yerr=flux_err, ls="none", capsize=1)
    # plt.plot(t, feature.model(t, band, *expected), "x", label="expected")
    # plt.plot(t, feature.model(t, band, *actual), "*", label="actual")
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=0.1)


def test_noisy_constant_temperature():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 5.0
    fall_time = 30.0
    Tmin = 5e3
    baselines = {b: 0.3 * amplitude + rng.exponential(scale=0.3 * amplitude) for b in band_wave_aa}

    expected = [reference_time, amplitude, rise_time, fall_time, Tmin, *baselines.values(), 1.0]

    feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=True, temperature="constant", bolometric="bazin")

    t = np.sort(rng.uniform(reference_time - 3 * rise_time, reference_time + 3 * fall_time, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))

    flux = feature.model(t, band, *expected)
    # S/N = 10 for minimum flux, scale for Poisson noise
    flux_err = np.sqrt(flux * np.min(flux)) / 10.0
    flux += rng.normal(0.0, flux_err)

    actual = feature(t, flux, sigma=flux_err, band=band)

    # import matplotlib.pyplot as plt
    # plt.scatter(t, flux, s=5, label="data")
    # plt.errorbar(t, flux, yerr=flux_err, ls="none", capsize=1)
    # plt.plot(t, feature.model(t, band, *expected), "x", label="expected")
    # plt.plot(t, feature.model(t, band, *actual), "*", label="actual")
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=0.1)


def test_noisy_constant_temperature_rising_only():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 5.0
    Tmin = 5e3
    baselines = {b: 0.3 * amplitude + rng.exponential(scale=0.3 * amplitude) for b in band_wave_aa}

    expected = [reference_time, amplitude, rise_time, Tmin, *baselines.values(), 1.0]

    feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline=True, temperature="constant", bolometric="sigmoid")

    t = np.sort(rng.uniform(reference_time - 3 * rise_time, reference_time + 3 * rise_time, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))

    flux = feature.model(t, band, *expected)
    # S/N = 5 for minimum flux, scale for Poisson noise
    flux_err = np.sqrt(flux * np.min(flux) / 5.0)
    flux += rng.normal(0.0, flux_err)

    actual = feature(t, flux, sigma=flux_err, band=band)

    # import matplotlib.pyplot as plt
    # plt.scatter(t, flux, s=5, label="data")
    # plt.errorbar(t, flux, yerr=flux_err, ls="none", capsize=1)
    # plt.plot(t, feature.model(t, band, *expected), "x", label="expected")
    # plt.plot(t, feature.model(t, band, *actual), "*", label="actual")
    # plt.legend()
    # plt.show()

    np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=0.1)
