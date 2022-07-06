from dataclasses import dataclass

import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.stats import sigmaclip

from ._base import BaseFeature


@dataclass()
class RedDwarfFit(BaseFeature):
    """Red dwarf flares fitting function.

    The function returns computed amplitude, fwhm, tpeak, chi2 and 7 parameters of the fit.

    - Depends on:  **time**, **magnitude**, **sigma**
    - Minimum number of observations: **7**
    - Number of features: **11**

    Note, that the function is developed to be used with fluxes, not magnitudes.

    Guadalupe Tovar Mendoza et al. 2022 [DOI:10.3847/1538-3881/ac6fe6](https://doi.org/10.3847/1538-3881/ac6fe6)
    """

    method: str = "scipy"

    def _eval(self, t, m, sigma=None):
        amplitude, fwhm, tpeak, background = RedDwarfFit._flare_params(t, m)
        norm_t = (t - tpeak) / fwhm

        if sigma is None:
            pass  # check if there is no sigma, do np.array(ones)

        if self.method == "scipy":
            popt = RedDwarfFit._scipy_optimize(t, m, amplitude, fwhm, tpeak, background)
        elif self.method == "minuit":
            popt = RedDwarfFit._minuit_optimize(t, m, sigma, amplitude, fwhm, tpeak, background)
        else:
            raise ValueError

        func = RedDwarfFit._func(amplitude)
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
        eq = lambda t, A, B, C, D1, D2, f1, background: amplitude * (
            background
            + (
                (1 / 2)
                * np.sqrt(np.pi)
                * A
                * C
                * f1
                * np.exp(-D1 * t + ((B / C) + (D1 * C / 2)) ** 2)
                * erfc(((B - t) / C) + (C * D1 / 2))
            )
            + (
                (1 / 2)
                * np.sqrt(np.pi)
                * A
                * C
                * (1 - f1)
                * np.exp(-D2 * t + ((B / C) + (D2 * C / 2)) ** 2)
                * erfc(((B - t) / C) + (C * D2 / 2))
            )
        )

        return eq

    @staticmethod
    def _scipy_optimize(t, m, amplitude, fwhm, tpeak, background):
        initial = np.array(
            [
                0.9687734504375167,
                -0.251299705922117,
                0.22675974948468916,
                0.15551880775110513,
                1.2150539528490194,
                0.12695865022878844,
                background / amplitude,
            ]
        )

        func = RedDwarfFit._func(amplitude)
        norm_t = (t - tpeak) / fwhm

        popt, _ = curve_fit(func, norm_t, m, p0=initial)

        return np.array(popt)

    @staticmethod
    def _minuit_optimize(t, m, sigma, amplitude, fwhm, tpeak, background):
        func = RedDwarfFit._func(amplitude)
        norm_t = (t - tpeak) / fwhm

        initial_dict = {
            "A": 0.9687734504375167,
            "B": -0.251299705922117,
            "C": 0.22675974948468916,
            "D1": 0.15551880775110513,
            "D2": 1.2150539528490194,
            "f1": 0.12695865022878844,
            "background": background / amplitude,
        }

        least_squares = LeastSquares(norm_t, m, sigma, func)
        fit = Minuit(least_squares, **initial_dict).migrad()

        popt = fit.values

        return np.array(popt)

    @staticmethod
    def model(t, params):
        amplitude, fwhm, tpeak, _, *popt = params
        func = RedDwarfFit._func(amplitude)
        predict = func(t, *popt)

        return predict

    @property
    def size(self):
        return 11


__all__ = ("RedDwarfFit",)
