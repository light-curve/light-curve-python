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
    common_bol_temp: List[str],
    common_temp_spec: List[str],
    bol: List[str],
    temp: List[str],
    spec: List[str],
    bands: Bands,
    with_baseline: bool,
):
    """Create an IntEnum class for RainbowFit parameters

    Parameters
    ----------
    cls_name : str
        Name of the class to create.
    common_bol_temp : list of str
        Common parameters for the bolometric and temperature models.
    common_temp_spec : list of str
        Common parameters for the spectral and temperature models.
    bol : list of str
        Bolometric model parameters, without common parameters.
    temp : list of str
        Temperature model parameters, without common parameters.
    spec : list of str
        Spectral model parameters.
    bands : list of str
        Unique list of bands in the dataset. It is used to generate baseline
        parameters when `with_baseline` is True.
    with_baseline : bool
        Whether to include baseline parameters, one per band in `bands`.
    """
    attributes = common_bol_temp + common_temp_spec + bol + temp + spec
    if with_baseline:
        baseline = list(map(baseline_parameter_name, bands.names))
        attributes += baseline

    enum = create_int_enum(cls_name, attributes)

    enum.all_common_bol_temp = common_bol_temp
    enum.common_bol_temp_idx = np.array([enum[attr] for attr in common_bol_temp])

    enum.all_common_temp_spec = common_temp_spec
    enum.common_temp_spec_idx = np.array([enum[attr] for attr in common_temp_spec])

    enum.bol = bol
    enum.bol_idx = np.array([enum[attr] for attr in enum.bol], dtype=np.int64)
    enum.all_bol = common_bol_temp + bol
    enum.all_bol_idx = np.array([enum[attr] for attr in enum.all_bol], dtype=np.int64)

    enum.temp = temp
    enum.temp_idx = np.array([enum[attr] for attr in enum.temp], dtype=np.int64)
    enum.all_temp = common_bol_temp + common_temp_spec + temp
    enum.all_temp_idx = np.array([enum[attr] for attr in enum.all_temp], dtype=np.int64)

    enum.spec = spec
    enum.spec_idx = np.array([enum[attr] for attr in enum.spec], dtype=np.int64)
    enum.all_spec = common_temp_spec + spec
    enum.all_spec_idx = np.array([enum[attr] for attr in enum.all_spec], dtype=np.int64)

    enum.with_baseline = with_baseline
    if with_baseline:
        enum.all_baseline = baseline
        # baseline_idx[i] is the parameter index of the baseline for band index i.
        # Bands are numbered in the same order as bands.names, so this array doubles
        # as a band_idx -> baseline_parameter_idx lookup table.
        enum.baseline_idx = np.array([enum[baseline_parameter_name(name)] for name in bands.names], dtype=np.int64)
        enum.baseline_parameter_name = staticmethod(baseline_parameter_name)
        enum.baseline_band_name = staticmethod(baseline_band_name)
        enum.lookup_baseline_idx_with_band_idx = enum.baseline_idx.__getitem__

    return enum
