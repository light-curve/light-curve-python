from dataclasses import dataclass
from typing import Tuple

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.special import erfc
from scipy.stats import sigmaclip

from ._base import BaseFeature


@dataclass()
class MendozaFit(BaseFeature):
    """White-light flares fitting function.

    The function returns computed amplitude, fwhm, tpeak, chi2 and 7 parameters of the Mendoza function developed
    for white-light flares:

    $$
    f(t) = f^* + \frac{\sqrt{\pi} A C}{2} \times (F_1 h(t, B, C, D_1) + F_2 h(t, B, C, D_2)),
    $$

    where $h(t, B, C, D) = \exp{(\alpha C D)} \times \text{erfc} (\alpha)$, $\alpha(t, C, D) = \frac12 C D + \frac{B
    - t}{C}$, where $erfc$ is $1-erf(t)$, t -- relative time, A -- amplitude, B -- position of the peak of the flare,
    C -- Gaussian heating timescale, $D_1$ -- rapid cooling phase timescale, $D_2$ -- slow cooling phase timescale,
    $F_2 \equiv 1 - F_1$ and  describe the relative importance of the exponential cooling terms.

    - Depends on:  **time**, **magnitude**, **sigma**
    - Minimum number of observations: **7**
    - Number of features: **11**

    Note, that the function is developed to be used with fluxes, not magnitudes.

    Guadalupe Tovar Mendoza et al. 2022 [DOI:10.3847/1538-3881/ac6fe6](https://doi.org/10.3847/1538-3881/ac6fe6)
    """

    def _eval(self, t, m, sigma=None):
        amplitude, fwhm, tpeak, background = MendozaFit._flare_params(t, m)
        norm_t = (t - tpeak) / fwhm

        popt = MendozaFit._minuit_optimize(t, m, sigma, amplitude, fwhm, tpeak, background)

        func = MendozaFit._func(amplitude)
        model = func(norm_t, *popt)
        chi2 = np.sum((m - model) ** 2) / (len(m) - 7)

        return np.concatenate((np.array([amplitude, fwhm, tpeak, chi2]), popt))

    @staticmethod
    def _flare_params(t, m):
        clipped = sigmaclip(m, low=3.0, high=3.0)[0]
        background = np.mean(clipped)
        clean_flux = m - background
        peak = np.argmax(clean_flux)
        amplitude = clean_flux[peak]
        tpeak = t[peak]

        condition = np.where(clean_flux[:peak] < 0.5 * clean_flux[peak])[0]
        if np.any(condition):
            left_idx = condition[-1]
        else:
            left_idx = peak

        left_t = t[left_idx] + (t[left_idx + 1] - t[left_idx]) * (0.5 * clean_flux[peak] - clean_flux[left_idx]) / (
            clean_flux[left_idx + 1] - clean_flux[left_idx]
        )

        condition = (np.where(clean_flux[peak:] < 0.5 * clean_flux[peak]) + peak)[0]
        if np.any(condition):
            right_idx = condition[0]
        else:
            right_idx = peak

        if right_idx == len(clean_flux) - 1:
            right_t = right_idx
        else:
            right_t = t[right_idx] + (t[right_idx + 1] - t[right_idx]) * (
                0.5 * clean_flux[peak] - clean_flux[right_idx]
            ) / (clean_flux[right_idx + 1] - clean_flux[right_idx])

        fwhm = right_t - left_t

        return amplitude, fwhm, tpeak, background

    @staticmethod
    def _func(amplitude):
        eq = lambda t, A1, A2, B, C, D1, D2, f1, background: amplitude * (
            background
            + (
                (1 / 2)
                * np.sqrt(np.pi)
                * A1
                * C
                * f1
                * np.exp(((1 / 2) * C * D1 + (B - t) / C) * C * D1)
                * erfc((1 / 2) * C * D1 + (B - t) / C)
            )
            + (
                (1 / 2)
                * np.sqrt(np.pi)
                * A2
                * C
                * (1 - f1)
                * np.exp(((1 / 2) * C * D2 + (B - t) / C) * C * D2)
                * erfc((1 / 2) * C * D2 + (B - t) / C)
            )
        )

        return eq

    @staticmethod
    def _minuit_optimize(t, m, sigma, amplitude, fwhm, tpeak, background):
        func = MendozaFit._func(amplitude)
        norm_t = (t - tpeak) / fwhm

        initial_dict = {
            "A1": 0.9687734504375167,
            "A2": 0.9687734504375167,
            "B": -0.251299705922117,
            "C": 0.22675974948468916,
            "D1": 0.15551880775110513,
            "D2": 1.2150539528490194,
            "f1": 0.12695865022878844,
            "background": background / amplitude,
        }

        initial_dict["A1"] = initial_dict["A1"] * np.exp(
            -(initial_dict["B"] ** 2) / (initial_dict["C"] ** 2)
            + (initial_dict["D1"] ** 2 * initial_dict["C"] ** 2) / 4
        )

        initial_dict["A2"] = initial_dict["A2"] * np.exp(
            -(initial_dict["B"] ** 2) / (initial_dict["C"] ** 2)
            + (initial_dict["D2"] ** 2 * initial_dict["C"] ** 2) / 4
        )

        least_squares = LeastSquares(norm_t, m, sigma, func)
        fit = Minuit(least_squares, **initial_dict).migrad()

        popt = fit.values

        return np.array(popt)

    @staticmethod
    def model(t, params):
        amplitude, fwhm, tpeak, _, *popt = params
        func = MendozaFit._func(amplitude)
        predict = func(t, *popt)

        return predict

    @property
    def size(self):
        return 11

    @property
    def names(self) -> Tuple[str, ...]:
        return ("amplitude", "fwhm", "tpeak", "chi2", "A1", "A2", "B", "C", "D1", "D2", "f1", "background")


__all__ = ("MendozaFit",)
