from dataclasses import dataclass

from ._base import BaseMultiBandFeature
from .median import Median


@dataclass()
class ColorOfMedian(BaseMultiBandFeature):
    """Difference of median magnitudes in two bands."""

    blue_band: str
    red_band: str

    @property
    def is_band_required(self) -> bool:
        return True

    @property
    def is_multiband_supported(self) -> bool:
        return True

    @property
    def size(self) -> int:
        return 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.median_feature = Median(bands=[self.blue_band, self.red_band])

    def _eval_and_fill(self, *, t, m, sigma, band, fill_value):
        median = self.median_feature._eval_and_fill(t=t, m=m, sigma=sigma, band=band, fill_value=fill_value)
        return median[0] - median[1]
