from dataclasses import dataclass
from typing import Collection, Union

import numpy as np

from light_curve.light_curve_ext import Extractor as _RustExtractor
from light_curve.light_curve_ext import _FeatureEvaluator as _RustBaseFeature

from ..dataclass_field import dataclass_field
from ._base import BaseSingleBandFeature


@dataclass()
class _PyExtractor(BaseSingleBandFeature):
    features: Collection[Union[BaseSingleBandFeature, _RustBaseFeature]] = dataclass_field(
        default_factory=list, kw_only=True
    )

    def _eval_single_band(self, t, m, sigma=None):
        raise NotImplementedError("_eval_single_band is missed for _PyExtractor")

    def _eval_and_fill_single_band(self, *, t, m, sigma, fill_value):
        return np.concatenate([np.atleast_1d(feature(t, m, sigma, fill_value=fill_value)) for feature in self.features])

    @property
    def size_single_band(self):
        return sum(
            feature.size if isinstance(feature, _RustBaseFeature) else feature.size_single_band
            for feature in self.features
        )


class Extractor:
    def __new__(cls, *args: Collection[Union[BaseSingleBandFeature, _RustBaseFeature]]):
        if len(args) > 0 and all(isinstance(feature, _RustBaseFeature) for feature in args):
            return _RustExtractor(*args)
        else:
            return _PyExtractor(features=args)


__all__ = ("Extractor",)
