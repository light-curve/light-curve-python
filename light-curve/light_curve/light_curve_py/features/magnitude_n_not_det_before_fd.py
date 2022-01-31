from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ._base import BaseFeature


@dataclass()
class MagnitudeNNotDetBeforeFd(BaseFeature):
    """Number of non detections before the first detection for measurements of the magnitude.

    Feature use a user-defined value to mark non-detections: measurements with sigma equal to this value
    considered as non detections. strictly_fainter flag allows counting non-detections with a strictly larger
    upper limit than the first detection magnitude (there is no such feature in the original article).

    - Depends on: **magnitude**
    - Minimum number of observations: **2**
    - Number of features: **1**

    Attributes
    ----------
    sigma_non_detection : float
        Sigma value to mark the non detections values, may not be NaN.
    strictly_fainter : bool
        Flag to determine if to find non-detections with strictly larger upper limit than the first detection magnitude.

    P. Sánchez-Sáez et al 2021, [DOI:10.3847/1538-3881/abd5c1](https://doi.org/10.3847/1538-3881/abd5c1)
    """

    sigma_non_detection: float = np.Inf
    strictly_fainter: bool = False

    def _eval(self, t, m, sigma=None, sorted=None, fill_value=None):
        detections = np.argwhere(sigma != self.sigma_non_detection).flatten()

        if len(detections) == len(m):
            raise ValueError("There is no any non-detections")

        first_detection_idx = detections[0]

        if self.strictly_fainter:
            detection_m = m[first_detection_idx]
            # magnitude upper limits should be larger than the first detection
            non_detection_less = np.count_nonzero(m[:first_detection_idx] > detection_m)
            return non_detection_less

        return first_detection_idx

    @property
    def names(self) -> Tuple[str]:
        return ("magn_n_non_detections_before_fd",)

    @property
    def descriptions(self) -> Tuple[str]:
        return ("number of non detections before the first detection for magnitudes",)

    @property
    def size(self):
        return 1


__all__ = ("MagnitudeNNotDetBeforeFd",)
