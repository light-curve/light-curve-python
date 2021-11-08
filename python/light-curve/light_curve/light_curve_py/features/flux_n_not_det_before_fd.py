import numpy as np

from dataclasses import dataclass
from ._base import BaseFeature


@dataclass()
class FluxNNotDetBeforeFd(BaseFeature):
    """Number of non-detections before the first detection for measurements of the flux.

    Feature use a user-defined signal to noise ratio to define non-detections and count their number before
    the first detection. Strictly_fainter flag allows counting non-detections with a strictly smaller upper limit
    than the first detection flux (there is no such feature in the original article).

    - Depends on: **flux**
    - Minimum number of observations: **2**
    - Number of features: **1**

    P. Sánchez-Sáez et al 2021, [DOI:10.3847/1538-3881/abd5c1](https://doi.org/10.3847/1538-3881/abd5c1)
    """

    signal_to_noise: float
    """Signal to noise ratio.
    """

    strictly_fainter: bool = False
    """Flag to determine if to find non-detections with strictly smaller upper limit than the first detection flux.
    """

    def _eval(self, t, m, sigma=None, sorted=None, fill_value=None):
        detections = np.argwhere(m > self.signal_to_noise * sigma).flatten()
        first_detection_idx = detections[0]

        if self.strictly_fainter:
            detection_m = m[first_detection_idx]
            upper_limits = sigma[:first_detection_idx] * self.signal_to_noise
            non_detection_less = np.count_nonzero(upper_limits < detection_m)
            return non_detection_less

        return first_detection_idx

    @property
    def names(self):
        return "flux_n_non_detections_before_fd"

    @property
    def descriptions(self):
        return "number of non detections before the first detection for fluxes"

    @property
    def size(self):
        return 1


__all__ = ("FluxNNotDetBeforeFd",)