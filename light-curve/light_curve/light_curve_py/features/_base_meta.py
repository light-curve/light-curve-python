from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Collection, Union

from light_curve.light_curve_ext import Extractor as _RustExtractor
from light_curve.light_curve_ext import _FeatureEvaluator as _RustBaseFeature

from ._base import BaseFeature
from .extractor import Extractor, _PyExtractor


@dataclass
class BaseMetaFeature(BaseFeature):
    features: Collection[Union[BaseFeature, _RustBaseFeature]] = ()
    extractor: Union[_RustExtractor, _PyExtractor] = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.extractor = Extractor(*self.features)

    @abstractmethod
    def transform(self, t, m, sigma):
        """Must return temporarily sorted arrays (t, m, sigma)"""
        pass

    def _eval(self, t, m, sigma=None):
        raise NotImplementedError("_eval is missed for BaseMetaFeature")

    def _eval_and_fill(self, t, m, sigma, fill_value):
        t, m, sigma = self.transform(t, m, sigma)
        return self.extractor._eval_and_fill(t, m, sigma, fill_value)

    @property
    def size(self):
        return self.extractor.size
