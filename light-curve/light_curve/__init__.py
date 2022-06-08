# Prevent isort to sort imports in this file
# isort: skip_file

# Import all Python features
from .light_curve_py import *

# Hide Python features with Rust equivalents
from .light_curve_ext import *

# Hide Rust Extractor with universal Python Extractor
from .light_curve_py import Extractor

from .light_curve_ext import __version__
from .light_curve_py.features._base import BaseFeature


__all__ = {"_FeatureEvaluator", "BaseFeature", "DmDt", "Extractor"}
__all__.update(c.__name__ for c in _FeatureEvaluator.__subclasses__() if c.__name__ in globals())
__all__.update(c.__name__ for c in BaseFeature.__subclasses__() if c.__name__ in globals())
__all__ = sorted(__all__)


__pdoc__ = {c.__name__: c.__pdoc__ for c in _FeatureEvaluator.__subclasses__()}
__pdoc__["_FeatureEvaluator"] = True
__pdoc__["light_curve_py"] = False
__pdoc__["light_curve_ext"] = False
__pdoc__["light_curve"] = False
