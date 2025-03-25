import numpy as np

from light_curve.light_curve_py import RainbowFit
from light_curve.light_curve_py.features.rainbow._scaler import MultiBandScaler


def test_noisy_with_baseline():
    rng = np.random.default_rng(0)

    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 5.0
    fall_time = 30.0
    Tmin = 5e3
    Tmax = 15e3
    t_color = 10
    baselines = {b: 0.3 * amplitude + rng.exponential(scale=0.3 * amplitude) for b in band_wave_aa}

    expected = [reference_time, amplitude, rise_time, fall_time, Tmin, Tmax, t_color, *baselines.values(), 1.0]

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

    np.testing.assert_allclose(feature.model(t, band, *expected), feature.model(t, band, *actual), rtol=0.1)


def test_noisy_all_functions_combination():
    rng = np.random.default_rng(0)
    band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0, "z": 9134.0}
    t = np.sort(rng.uniform(59985.0, 60090.0, 1000))
    band = rng.choice(list(band_wave_aa), size=len(t))

    bazin_parameters = [
        60000.0,  # reference_time
        1.0,  # amplitude
        5.0,  # rise_time
        30.0,  # fall_time
    ]

    sigmoid_parameters = [
        60000.0,  # reference_time
        1.0,  # amplitude
        5.0,  # rise_time
    ]

    linexp_parameters = [
        60000.0,  # reference_time
        1,  # amplitude
        -20,  # rise_time
    ]

    doublexp_parameters = [60000.0, 10, 5, 10, 0.1]  # reference_time  # amplitude  # time1  # time2  # p

    bolometric_names = ["bazin", "sigmoid", "linexp", "doublexp"]
    bolometric_params = [bazin_parameters, sigmoid_parameters, linexp_parameters, doublexp_parameters]

    Tsigmoid_parameters = [5e3, 15e3, 10]  # Tmin  # Tmax  # t_color

    constant_parameters = [1e4]  # T

    temperature_names = ["constant", "sigmoid"]
    temperature_params = [constant_parameters, Tsigmoid_parameters]

    for idx_b in range(len(bolometric_names)):
        for idx_t in range(len(temperature_names)):

            expected = [*bolometric_params[idx_b], *temperature_params[idx_t], 1.0]

            feature = RainbowFit.from_angstrom(
                band_wave_aa,
                with_baseline=False,
                temperature=temperature_names[idx_t],
                bolometric=bolometric_names[idx_b],
            )

            flux = feature.model(t, band, *expected)

            # The linexp function can reach unphysical negative flux values
            protected_flux = np.where(flux > 1e-3, flux, 1e-3)

            # S/N = 10 for minimum flux, scale for Poisson noise
            flux_err = np.sqrt(protected_flux * np.min(protected_flux)) / 10.0
            flux += rng.normal(0.0, flux_err)

            actual = feature(t, flux, sigma=flux_err, band=band)

            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.scatter(t, flux, s=5, label="data")
            # plt.errorbar(t, flux, yerr=flux_err, ls="none", capsize=1)
            # plt.plot(t, feature.model(t, band, *expected), "x", label="expected")
            # plt.plot(t, feature.model(t, band, *actual), "*", label="actual")
            # plt.ylim(-.05, flux.max()+0.1)
            # plt.legend()
            # plt.show()

            # The first test might be too rigid. The second test allow for good local minima to be accepted
            np.testing.assert_allclose(actual[:-1], expected[:-1], rtol=0.1)

            # If either the absolute or the relative test passes, it is accepted.
            # It prevents linexp, which include a flat exactly 0 baseline to not pass the test because
            # of very minor parameter differences that lead to a major relative difference.
            np.testing.assert_allclose(
                feature.model(t, band, *expected), feature.model(t, band, *actual), rtol=0.1, atol=0.1, strict=False
            )


def test_scaler_from_flux_list_input():
    "https://github.com/light-curve/light-curve-python/issues/492"
    # Was failing
    scaler1 = MultiBandScaler.from_flux(
        flux=[1.0, 2.0, 3.0, 4.0], band=np.array(["g", "r", "g", "r"]), with_baseline=True
    )
    # Was not failing, but was wrong
    scaler2 = MultiBandScaler.from_flux(flux=[1.0, 2.0, 3.0, 4.0], band=["g", "r", "g", "r"], with_baseline=True)
    assert scaler1 == scaler2
