# import matplotlib.pyplot as plt
import numpy as np

from light_curve.light_curve_py import RainbowFit
from light_curve.light_curve_py.features.rainbow._scaler import MultiBandScaler


def test_noisy_with_baseline():

    rng = np.random.default_rng(0)

    band_wave_aa = {
        "g": 4770.0,
        "r": 6231.0,
        "i": 7625.0,
        "z": 9134.0,
    }

    reference_time = 60000.0
    amplitude = 1.0
    rise_time = 5.0
    fall_time = 30.0

    Tmin = 5e3
    Tmax = 15e3
    t_color = 10

    lambda_scale = 500.0

    baselines = {b: 0.3 * amplitude + rng.exponential(scale=0.3 * amplitude) for b in band_wave_aa}

    expected = [
        reference_time,
        amplitude,
        rise_time,
        fall_time,
        Tmin,
        Tmax,
        t_color,
        lambda_scale,
        *baselines.values(),
        1.0,
    ]

    feature = RainbowFit.from_angstrom(
        band_wave_aa,
        with_baseline=True,
        temperature="sigmoid",
        bolometric="bazin",
        spectral="blanketed",
    )

    t = np.sort(
        rng.uniform(
            reference_time - 3 * rise_time,
            reference_time + 3 * fall_time,
            1000,
        )
    )

    band = rng.choice(list(band_wave_aa), size=len(t))

    flux = feature.model(t, band, *expected[:-1])

    protected_flux = np.where(flux > 1e-3, flux, 1e-3)

    flux_err = np.sqrt(protected_flux * np.min(protected_flux)) / 10.0

    flux += rng.normal(0.0, flux_err)

    actual = feature(
        t,
        flux,
        sigma=flux_err,
        band=band,
    )

    """
    colors = {
        "g": "blue",
        "r": "green",
        "i": "orange",
        "z": "red",
    }

    plt.figure(figsize=(12, 5))

    for b in band_wave_aa:
        mask = band == b

        plt.errorbar(
            t[mask],
            flux[mask],
            yerr=flux_err[mask],
            ls="none",
            fmt=".",
            alpha=0.3,
            color=colors[b],
        )

        plt.plot(
            t[mask],
            feature.model(t[mask], band[mask], *expected[:-1]),
            color=colors[b],
            linewidth=2,
            label=f"{b} expected",
        )

        plt.plot(
            t[mask],
            feature.model(t[mask], band[mask], *actual[:-1]),
            "--",
            color=colors[b],
            linewidth=2,
            label=f"{b} fitted",
        )

    plt.legend()
    plt.title("Blanketed model with baseline")
    plt.xlabel("time")
    plt.ylabel("flux")
    plt.show()
    """

    np.testing.assert_allclose(
        feature.model(t, band, *expected[:-1]),
        feature.model(t, band, *actual[:-1]),
        rtol=0.1,
    )


def test_noisy_all_functions_combination():

    rng = np.random.default_rng(0)

    band_wave_aa = {
        "g": 4770.0,
        "r": 6231.0,
        "i": 7625.0,
        "z": 9134.0,
    }

    t = np.sort(rng.uniform(59985.0, 60090.0, 1000))

    band = rng.choice(list(band_wave_aa), size=len(t))

    # ======================================================
    # Bolometric models
    # ======================================================

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

    doublexp_parameters = [
        60000.0,  # reference_time
        10,  # amplitude
        5,  # time1
        10,  # time2
        0.1,  # p
    ]

    bolometric_names = [
        "bazin",
        "sigmoid",
        "linexp",
        "doublexp",
    ]

    bolometric_params = [
        bazin_parameters,
        sigmoid_parameters,
        linexp_parameters,
        doublexp_parameters,
    ]

    # ======================================================
    # Temperature models
    # ======================================================

    Tsigmoid_parameters = [
        5e3,  # Tmin
        15e3,  # Tmax
        10,  # t_color
    ]

    constant_parameters = [
        1e4,  # Temperature
    ]

    temperature_names = [
        "constant",
        "sigmoid",
    ]

    temperature_params = [
        constant_parameters,
        Tsigmoid_parameters,
    ]

    # ======================================================
    # Spectral models
    # ======================================================

    BB_parameters = []

    UV_BB_parameters = [
        800.0,  # lambda_scale
    ]

    spectral_names = [
        "planck",
        "blanketed",
    ]

    spectral_params = [
        BB_parameters,
        UV_BB_parameters,
    ]

    # ======================================================
    # Loop over all combinations
    # ======================================================

    for idx_b in range(len(bolometric_names)):
        for idx_t in range(len(temperature_names)):
            for idx_s in range(len(spectral_names)):
                expected = [
                    *bolometric_params[idx_b],
                    *temperature_params[idx_t],
                    *spectral_params[idx_s],
                    1.0,
                ]

                feature = RainbowFit.from_angstrom(
                    band_wave_aa,
                    with_baseline=False,
                    temperature=temperature_names[idx_t],
                    bolometric=bolometric_names[idx_b],
                    spectral=spectral_names[idx_s],
                )

                flux = feature.model(t, band, *expected[:-1])

                protected_flux = np.where(
                    flux > 1e-3,
                    flux,
                    1e-3,
                )

                flux_err = np.sqrt(protected_flux * np.min(protected_flux)) / 10.0

                flux += rng.normal(0.0, flux_err)

                actual = feature(
                    t,
                    flux,
                    sigma=flux_err,
                    band=band,
                )

                """
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 4))

                plt.scatter(
                    t,
                    flux,
                    s=5,
                    alpha=0.4,
                    label="data",
                )

                plt.plot(
                    t,
                    feature.model(t, band, *expected[:-1]),
                    "x",
                    label="expected",
                )

                plt.plot(
                    t,
                    feature.model(t, band, *actual[:-1]),
                    ".",
                    label="actual",
                )

                plt.title(f"{bolometric_names[idx_b]} + {temperature_names[idx_t]} + {spectral_names[idx_s]}")

                plt.legend()
                plt.show()
                """

                """ Too strict requirement. Parameters of complex models are too degenerate.
                np.testing.assert_allclose(
                    actual[:-1],
                    expected[:-1],
                    rtol=0.1,
                )
                """

                np.testing.assert_allclose(
                    feature.model(t, band, *expected[:-1]),
                    feature.model(t, band, *actual[:-1]),
                    rtol=0.1,
                    atol=0.1,
                    strict=False,
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
