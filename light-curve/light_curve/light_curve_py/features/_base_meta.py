from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Collection, Union

from light_curve.light_curve_ext import Extractor as _RustExtractor
from light_curve.light_curve_ext import _FeatureEvaluator as _RustBaseFeature

from ..dataclass_field import dataclass_field
from ._base import BaseSingleBandFeature
from .extractor import Extractor, _PyExtractor


@dataclass
class BaseMetaSingleBandFeature(BaseSingleBandFeature):
    features: Collection[Union[BaseSingleBandFeature, _RustBaseFeature]] = dataclass_field(
        default_factory=list, kw_only=True
    )
    extractor: Union[_RustExtractor, _PyExtractor] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.extractor = Extractor(*self.features)

    @abstractmethod
    def transform(self, *, t, m, sigma):
        """Must return temporarily sorted arrays (t, m, sigma)"""
        pass

    def _eval_single_band(self, t, m, sigma=None):
        raise NotImplementedError("_eval_single_band is missed for BaseMetaFeature")

    def _eval_and_fill_single_band(self, *, t, m, sigma, fill_value):
        t, m, sigma = self.transform(t=t, m=m, sigma=sigma)
        return self.extractor._eval_and_fill_single_band(t=t, m=m, sigma=sigma, fill_value=fill_value)

    @property
    def size_single_band(self):
        if isinstance(self.extractor, _RustExtractor):
            return self.extractor.size
        return self.extractor.size_single_band
