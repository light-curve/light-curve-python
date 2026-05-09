from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ..light_curve_py.warnings import warn_experimental
from .reduction import InputTensors, Reduction, reduction_from_str


@dataclass
class AstromerInputs(InputTensors):
    """Input tensors for Astromer-family models.

    ``times`` and ``input`` are float32 arrays of shape
    ``(n_windows, seq_size, 1)`` ready for ONNX inference.  ``mask`` is the
    same validity information cast to float32 for the model (1 = valid,
    0 = padded), shape ``(n_windows, seq_size, 1)``.  ``bool_mask`` (inherited)
    is the boolean equivalent, shape ``(n_windows, seq_size)``.
    """

    times: np.ndarray = field(kw_only=True)
    input: np.ndarray = field(kw_only=True)
    mask: np.ndarray = field(kw_only=True)


class Dim(IntEnum):
    """Axis indices for the 4-D embedding array ``(BAND, SUBSAMPLE, SEQUENCE, VALUE)``."""

    BAND = 0
    SUBSAMPLE = 1
    SEQUENCE = 2
    VALUE = 3


class EmbeddingSession(ABC):
    """Abstract base for ONNX-backed embedding models.

    Subclasses implement :meth:`preprocess_lc` (convert raw arrays to model
    tensors) and :meth:`predict_tensors` (run the session and return embeddings).

    Parameters
    ----------
    session :
        An ``onnxruntime.InferenceSession`` (or any object with a compatible
        ``.run()`` interface).
    reduction : str, list of str, or Reduction
        Strategy for mapping variable-length light curves to fixed-length
        sequences.
    time_red_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`reduction_from_str`
        when ``reduction`` is given as a string.
    """

    seq_size: int

    def __init__(
        self,
        session,
        *,
        reduction: str | list[str] | Reduction,
        time_red_kwargs: dict[str, object] | None = None,
    ) -> None:
        warn_experimental(
            f"{self.__class__.__module__}.{self.__class__.__name__} is experimental and may change in future versions"
        )
        self.session = session

        if time_red_kwargs is None:
            time_red_kwargs = {}
        if not isinstance(reduction, Reduction):
            reduction = reduction_from_str(reduction, **time_red_kwargs)
        self.reduction = reduction

    @abstractmethod
    def __call__(self, *arrays: ArrayLike) -> np.ndarray:
        inputs = self.preprocess_lc(*arrays)
        embedding = self.predict_tensors(inputs)
        return embedding

    @abstractmethod
    def preprocess_lc(self, *arrays: ArrayLike) -> InputTensors:
        """Convert raw light curve arrays to model input tensors.

        Parameters
        ----------
        *arrays : array-like
            Raw light curve arrays (e.g. time, magnitude).

        Returns
        -------
        InputTensors
            Tensors ready to be passed to :meth:`predict_tensors`.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_tensors(self, tensors: InputTensors) -> np.ndarray:
        """Run the ONNX session on pre-processed tensors and return embeddings.

        Parameters
        ----------
        tensors : InputTensors
            Pre-processed model inputs as returned by :meth:`preprocess_lc`.

        Returns
        -------
        np.ndarray
            Embedding array with shape depending on the model and time reduction.
        """
        raise NotImplementedError


class SingleBandModel(EmbeddingSession, ABC):
    """Embedding model that processes one photometric band at a time.

    When ``bands`` is ``None`` the full light curve is treated as a single band.
    When ``bands`` is provided, the light curve is split by band label, each band
    is embedded independently, and the results are concatenated along
    :attr:`Dim.BAND`.

    Parameters
    ----------
    session :
        ONNX inference session.
    bands : sequence of str or int, optional
        Ordered band labels to embed.  ``None`` treats the whole light curve as
        one band.
    reduction : str, list of str, or Reduction
        Windowing / subsampling strategy.  Defaults to
        ``"non-overlapping-windows"``.
    time_red_kwargs : dict, optional
        Extra kwargs forwarded to :func:`reduction_from_str`.
    """

    def __init__(
        self,
        session,
        *,
        bands: Sequence[str | int] | None = None,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        time_red_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            session,
            reduction=reduction,
            time_red_kwargs=time_red_kwargs,
        )
        self.bands = bands

    def __call__(self, time: ArrayLike, mag: ArrayLike, band: ArrayLike | None = None) -> np.ndarray:
        """Embed a light curve.

        Parameters
        ----------
        time : array-like, shape ``(n,)``
            Observation times.
        mag : array-like, shape ``(n,)``
            Magnitudes.
        band : array-like, shape ``(n,)``, optional
            Band labels, required when ``self.bands`` is not ``None``.

        Returns
        -------
        np.ndarray, shape ``(n_bands, n_subsamples, seq_size, embed_dim)``
            Embedding tensor.  ``n_bands`` is 1 when ``self.bands`` is ``None``.

        Raises
        ------
        ValueError
            If ``band`` is provided but ``self.bands`` is ``None``, or vice versa.
        """
        if (band is None) != (self.bands is None):
            raise ValueError(
                f"band array must be provided with values specified by `bands` class constructor argument: {self.bands}"
            )

        if self.bands is None:
            embed = super().__call__(time, mag)
            return np.expand_dims(embed, axis=Dim.BAND)

        embeddings = []
        for band_name in self.bands:
            band_idx = band == band_name
            embed_band = super().__call__(time[band_idx], mag[band_idx])
            embeddings.append(np.expand_dims(embed_band, axis=Dim.BAND))
        return np.concatenate(embeddings, axis=Dim.BAND)
