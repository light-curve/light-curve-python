from enum import IntEnum
from typing import Iterable, List

import numpy as np

from light_curve.light_curve_py.features.rainbow._bands import Bands

__all__ = ["create_parameters_class"]


def baseline_parameter_name(band: str) -> str:
    return f"baseline_{band}"


def baseline_band_name(name: str) -> str:
    if name.startswith("baseline_"):
        return name[len("baseline_") :]

    return None


def create_int_enum(cls_name: str, attributes: Iterable[str]):
    return IntEnum(cls_name, {attr: i for i, attr in enumerate(attributes)})


def create_parameters_class(
    cls_name: str,
    *,
    common: List[str],
    bol: List[str],
    temp: List[str],
    bands: Bands,
    with_baseline: bool,
):
    """Create an IntEnum class for RainbowFit parameters

    Parameters
    ----------
    cls_name : str
        Name of the class to create
    common : list of str
        Common parameters for both bolometric and temperature models
    bol : list of str
        Bolometric model parameters, without common parameters
    temp : list of str
        Temperature model parameters, without common parameters
    bands : list of str
        Unique list of bands in the dataset. It is used to generate baseline
        parameters when `with_baseline` is True.
    with_baseline : bool
        Whether to include baseline parameters, one per band in `bands`.
    """
    attributes = common + bol + temp
    if with_baseline:
        baseline = list(map(baseline_parameter_name, bands.names))
        attributes += baseline

    enum = create_int_enum(cls_name, attributes)

    enum.all_common = common
    enum.common_idx = np.array([enum[attr] for attr in common])

    enum.bol = bol
    enum.bol_idx = np.array([enum[attr] for attr in enum.bol])
    enum.all_bol = common + bol
    enum.all_bol_idx = np.array([enum[attr] for attr in enum.all_bol])

    enum.temp = temp
    enum.temp_idx = np.array([enum[attr] for attr in enum.temp])
    enum.all_temp = common + temp
    enum.all_temp_idx = np.array([enum[attr] for attr in enum.all_temp])

    enum.with_baseline = with_baseline
    if with_baseline:
        enum.all_baseline = baseline
        enum.baseline_idx = np.array([enum[attr] for attr in enum.all_baseline])
        enum.baseline_parameter_name = staticmethod(baseline_parameter_name)
        enum.baseline_band_name = staticmethod(baseline_band_name)

        band_idx_to_baseline_idx = {
            band_idx: enum[baseline_parameter_name(band_name)] for band_idx, band_name in zip(bands.index, bands.names)
        }
        enum.lookup_baseline_idx_with_band_idx = np.vectorize(band_idx_to_baseline_idx.get, otypes=[int])

    return enum
