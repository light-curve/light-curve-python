from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..dataclass_field import dataclass_field
from ._base import BaseSingleBandFeature


@dataclass()
class FluxNNotDetBeforeFd(BaseSingleBandFeature):
    """Number of non-detections before the first detection for measurements of the flux.

    Feature use a user-defined signal to noise ratio to define non-detections and count their number before
    the first detection. strictly_fainter flag allows counting non-detections with a strictly smaller upper limit
    than the first detection flux (there is no such feature in the original article).

    - Depends on: **flux**
    - Minimum number of observations: **2**
    - Number of features: **1**

    Attributes
    ----------
    signal_to_noise : float
        Signal to noise ratio.
    strictly_fainter : bool
        Flag to determine if to find non-detections with strictly smaller upper limit than the first detection flux.

    P. Sánchez-Sáez et al 2021, [DOI:10.3847/1538-3881/abd5c1](https://doi.org/10.3847/1538-3881/abd5c1)
    """

    signal_to_noise: float = dataclass_field(default=5.0, kw_only=True)
    strictly_fainter: bool = dataclass_field(default=False, kw_only=True)

    def _eval_single_band(self, t, m, sigma=None):
        detections = np.argwhere(m > self.signal_to_noise * sigma).flatten()

        if len(detections) == len(m):
            raise ValueError("There is no any non-detections")

        first_detection_idx = detections[0]

        if self.strictly_fainter:
            detection_m = m[first_detection_idx]
            upper_limits = sigma[:first_detection_idx] * self.signal_to_noise
            non_detection_less = np.count_nonzero(upper_limits < detection_m)
            return non_detection_less

        return first_detection_idx

    @property
    def names(self) -> Tuple[str]:
        return ("flux_n_non_detections_before_fd",)

    @property
    def descriptions(self) -> Tuple[str]:
        return ("number of non detections before the first detection for fluxes",)

    @property
    def size_single_band(self) -> int:
        return 1


__all__ = ("FluxNNotDetBeforeFd",)
