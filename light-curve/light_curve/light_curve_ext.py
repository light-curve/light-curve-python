from .light_curve import *


# A hack to bypass the lack of package support in PyO3
# https://github.com/PyO3/pyo3/issues/1517#issuecomment-808664021
def __register_submodules():
    import sys

    sys.modules["light_curve.light_curve_ext.ln_prior"] = ln_prior


__register_submodules()
