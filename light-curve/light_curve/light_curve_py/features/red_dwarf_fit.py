import numpy as np
from scipy.special import erfc
from scipy.stats import chisquare, sigmaclip

from ._base import BaseFeature


class RedDwarfFit(BaseFeature):
    """Red dwarf flares fitting function.

    - Depends on:  **time**, **magnitude**
    - Minimum number of observations: **2**
    - Number of features: **4**

    Note, that the function is developed to be used with fluxes, not magnitudes.

    Guadalupe Tovar Mendoza et al. 2022 [DOI:10.3847/1538-3881/ac6fe6](https://doi.org/10.3847/1538-3881/ac6fe6)
    """

    def _eval(self, t, m, sigma=None):
        amplitude, fwhm, tpeak = RedDwarfFit._flare_params(t, m)
        model = self.fit(t, amplitude, fwhm, tpeak)
        chi2 = np.sum((m - model) ** 2) / (len(m) - 1)

        return np.array([amplitude, fwhm, tpeak, chi2])

    @staticmethod
    def _flare_params(t, m):
        clipped = sigmaclip(m, low=3.0, high=3.0)[0]
        background = np.mean(clipped)
        clean_flux = m - background
        peak = np.argmax(clean_flux)
        amplitude = clean_flux[peak]
        tpeak = t[peak]

        left_idx = np.where(clean_flux[:peak] < 0.5 * clean_flux[peak])[0][-1]
        left_t = t[left_idx] + (t[left_idx + 1] - t[left_idx]) * (0.5 * clean_flux[peak] - clean_flux[left_idx]) / (
            clean_flux[left_idx + 1] - clean_flux[left_idx]
        )

        right_idx = (np.where(clean_flux[peak:] < 0.5 * clean_flux[peak]) + peak)[0][0]
        right_t = t[right_idx] + (t[right_idx + 1] - t[right_idx]) * (
            0.5 * clean_flux[peak] - clean_flux[right_idx]
        ) / (clean_flux[right_idx + 1] - clean_flux[right_idx])

        fwhm = right_t - left_t

        return amplitude, fwhm, tpeak

    @staticmethod
    def fit(t, amplitude, fwhm, tpeak):
        A, B, C, D1, D2, f1 = [
            0.9687734504375167,
            -0.251299705922117,
            0.22675974948468916,
            0.15551880775110513,
            1.2150539528490194,
            0.12695865022878844,
        ]

        t_new = (t - tpeak) / fwhm
        f2 = 1 - f1

        eq = (
            (1 / 2)
            * np.sqrt(np.pi)
            * A
            * C
            * f1
            * np.exp(-D1 * t_new + ((B / C) + (D1 * C / 2)) ** 2)
            * erfc(((B - t_new) / C) + (C * D1 / 2))
        ) + (
            (1 / 2)
            * np.sqrt(np.pi)
            * A
            * C
            * f2
            * np.exp(-D2 * t_new + ((B / C) + (D2 * C / 2)) ** 2)
            * erfc(((B - t_new) / C) + (C * D2 / 2))
        )

        return eq * amplitude

    @property
    def size(self):
        return 4


__all__ = ("RedDwarfFit",)
