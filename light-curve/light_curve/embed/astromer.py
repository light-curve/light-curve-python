from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import ArrayLike

from light_curve.embed.input_tensors import InputTensors
from light_curve.embed.model import SingleBandModel
from light_curve.embed.reduction import Reduction

if TYPE_CHECKING:
    from typing import Self


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
    mask_in: np.ndarray = field(kw_only=True)


class _AstromerModel(SingleBandModel):
    """Internal base class for Astromer-family embedding models.

    Provides shared preprocessing (per-window zero-mean normalisation) and
    ONNX inference logic for all Astromer variants.  Concrete subclasses set
    :attr:`hf_repo` and, if needed, override the default ``reduction``.

    Output shape
    ------------
    :meth:`__call__` always returns a 4-D array
    ``(n_bands, n_subsamples, seq_size, embed_dim)`` for consistency across
    models and windowing strategies:

    * ``n_bands`` — number of photometric bands; 1 when ``bands`` is ``None``.
    * ``n_subsamples`` — windows produced by the time reduction (1 for
      :class:`NonOverlappingWindows`, which averages all windows).
    * ``seq_size`` — 1 for the ``"mean"`` and ``"max"`` outputs (aggregated
      over the sequence); equal to the model's sequence length for ``"sequence"``.
    * ``embed_dim`` — embedding dimension (256 for all Astromer models).

    Use ``embedding.squeeze()`` to collapse unit dimensions when the full
    4-D layout is not needed.
    """

    seq_size: int = 200
    dtype: type = np.float32
    hf_filename: str
    model_outputs: frozenset[str] = frozenset({"mean", "max", "sequence"})

    def __init__(
        self,
        session,
        *,
        output: str = "mean",
        bands: Sequence[str | int] | None = None,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        reduction_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            session,
            bands=bands,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
        )
        if output not in self.model_outputs:
            raise ValueError(f"Unknown output '{output}'. Must be one of: {', '.join(sorted(self.model_outputs))}")
        self.output = output

    def __call__(self, time: ArrayLike, mag: ArrayLike, band: ArrayLike | None = None) -> np.ndarray:
        """Embed a light curve.

        Parameters
        ----------
        Parameters
        ----------
        time : array-like, shape ``(n,)``
            Observation times in days (e.g. MJD).
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
        return super().__call__(time, mag, band=band)

    @classmethod
    def from_hf(
        cls,
        output: str = "mean",
        *,
        bands: Sequence[str | int] | None = None,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        reduction_kwargs: dict[str, object] | None = None,
        ort_session_kwargs: dict[str, object] | None = None,
    ) -> Self:
        """Load a model from the HuggingFace Hub.

        Downloads (and caches) the ONNX model file, creates an
        ``onnxruntime.InferenceSession``, and returns a ready-to-use instance.
        Only the requested output is computed at inference time — onnxruntime
        prunes the unused computation graph automatically.

        Parameters
        ----------
        output : str, optional
            Named ONNX output to return.  One of:

            * ``"mean"`` (default) — masked mean pooling over valid timesteps,
              output shape ``(bands, reductions, 1, 256)``
            * ``"max"`` — masked max pooling over valid timesteps,
              output shape ``(bands, reductions, 1, 256)``
            * ``"sequence"`` — per-timestep embeddings (no aggregation),
              output shape ``(bands, reductions, 200, 256)``

        bands : sequence of str or int or None, optional
            Ordered band labels to embed.  ``None`` (default) treats the whole
            light curve as one band.
        reduction : str, list of str, or Reduction, optional
            Windowing / subsampling strategy.  Defaults to
            ``"non-overlapping-windows"``.
        reduction_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :func:`reduction_from_str`
            when ``reduction`` is given as a string.
        ort_session_kwargs : dict or None, optional
            Additional keyword arguments forwarded to ``onnxruntime.InferenceSession``:
            "sess_options", "providers", "provider_options".

        Returns
        -------
        instance of the calling class
            Instance with a live ONNX inference session.

        Raises
        ------
        ValueError
            If ``output`` is not one of the recognised output names.
        ImportError
            If ``huggingface_hub`` is not installed.
        ImportError
            If no ``onnxruntime`` variant is installed.
        """
        return super().from_hf(
            cls.hf_filename,
            output=output,
            bands=bands,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
            ort_session_kwargs=ort_session_kwargs,
        )

    def preprocess_lc(
        self,
        time: ArrayLike,
        mag: ArrayLike,
    ) -> AstromerInputs:
        """Preprocess a light curve into Astromer model input tensors.

        Each window is independently zero-mean normalised (time and magnitude)
        using only the valid (non-padded) observations.  Normalisation is
        computed in the original input precision before casting to ``float32``
        to avoid precision loss for large values such as MJD timestamps.

        Parameters
        ----------
        time : array-like, shape ``(n,)``
            Observation times (e.g. MJD).
        mag : array-like, shape ``(n,)``
            Magnitudes.

        Returns
        -------
        AstromerInputs
            ``times`` and ``input`` are per-window zero-mean arrays of shape
            ``(n_windows, seq_size, 1)`` in float32; ``mask`` is 1 for valid
            observations and 0 for zero-padded positions, same shape and dtype.
        """
        time, mag, mask = self.reduction.preprocess_lc(time, mag, seq_size=self.seq_size)

        bool_mask = mask  # (n_windows, seq_size), boolean
        n_valid = mask.sum(axis=-1, keepdims=True)
        time_mean = (time * mask).sum(axis=-1, keepdims=True) / n_valid
        mag_mean = (mag * mask).sum(axis=-1, keepdims=True) / n_valid
        time = np.where(mask, time - time_mean, time).astype(self.dtype)
        mag = np.where(mask, mag - mag_mean, mag).astype(self.dtype)
        mask = mask.astype(self.dtype)

        idx = (..., np.newaxis)
        return AstromerInputs(times=time[idx], input=mag[idx], mask_in=mask[idx], bool_mask=bool_mask)

    def predict_tensors(self, tensors: AstromerInputs) -> np.ndarray:
        """Run the ONNX model on pre-processed tensors and return reduced embeddings.

        Parameters
        ----------
        tensors : AstromerInputs
            As returned by :meth:`preprocess_lc`.

        Returns
        -------
        np.ndarray, shape ``(n_subsamples, seq_size, embed_dim)``
            Embeddings after applying the time reduction's aggregation.
            For aggregated models (mean / max) ``seq_size`` is 1.
        """
        (raw_embedding,) = self.session.run(
            [self.output],
            # {"input": tensors.input, "times": tensors.times, "mask_in": tensors.mask},
            tensors.asdict(),
        )

        # Aggregated outputs (mean / max) have shape (n_windows, embed_dim); add SEQUENCE=1 axis
        if raw_embedding.ndim == 2:
            raw_embedding = np.expand_dims(raw_embedding, axis=1)

        return self.reduction.reduce_embeddings(raw_embedding, tensors, output=self.output)


class Astromer1(_AstromerModel):
    """Astromer 1 embedding model.

    Transformer encoder pretrained on MACHO R-band light curves via masked
    magnitude prediction.  Accepts single-band photometry
    and returns a 256-dimensional embedding (2 layers, 4 attention heads).

    The ONNX model is hosted on HuggingFace at
    ``https://huggingface.co/light-curve/astromer1`` (``astromer1.onnx``).
    Three named outputs are available; select with the ``output`` parameter:

    * ``"mean"`` (default) — masked mean pooling → shape ``(batch, 256)``
    * ``"max"`` — masked max pooling → shape ``(batch, 256)``
    * ``"sequence"`` — per-timestep features → shape ``(batch, 200, 256)``

    Use :meth:`from_hf` to download and load the model directly.

    Model license
    -------------
    MIT.

    References
    ----------
    Donoso-Oliva et al. (2023), *ASTROMER: A transformer-based embedding for
    the representation of light curves*, A&A 670, A54.
    https://ui.adsabs.harvard.edu/abs/2023A%26A...670A..54D/abstract

    Parameters
    ----------
    session :
        ONNX inference session for the Astromer 1 model file.
    output : str, optional
        Which named output to return: ``"mean"``, ``"max"``, or ``"sequence"``.
        Defaults to ``"mean"``.
    bands : sequence of str or int, optional
        Band labels.  ``None`` (default) treats the whole light curve as one
        band.
    reduction : str, list of str, or Reduction
        Windowing strategy.  Defaults to :class:`NonOverlappingWindows`.
    """

    hf_repo: str = "light-curve/astromer1"
    hf_filename: str = "astromer1.onnx"


class Astromer2(_AstromerModel):
    """Astromer 2 embedding model.

    Pretrained on 1.5 million MACHO light curves. Accepts
    single-band photometry and returns a 256-dimensional embedding.

    The ONNX model is hosted on HuggingFace at
    ``https://huggingface.co/light-curve/astromer2`` (``astromer2.onnx``).
    Three named outputs are available; select with the ``output`` parameter:

    * ``"mean"`` (default) — masked mean pooling → shape ``(batch, 256)``
    * ``"max"`` — masked max pooling → shape ``(batch, 256)``
    * ``"sequence"`` — per-timestep features → shape ``(batch, 200, 256)``

    Use :meth:`from_hf` to download and load the model directly.

    Model license
    -------------
    MIT.

    References
    ----------
    Donoso-Oliva et al. (2026), *Generalizing across astronomical surveys:
    Few-shot light curve classification with Astromer 2*, A&A 707, A170.
    https://ui.adsabs.harvard.edu/abs/2026A%26A...707A.170D/abstract

    Parameters
    ----------
    session :
        ONNX inference session for the Astromer 2 model file.
    output : str, optional
        Which named output to return: ``"mean"``, ``"max"``, or ``"sequence"``.
        Defaults to ``"mean"``.
    bands : sequence of str or int, optional
        Band labels.  ``None`` (default) treats the whole light curve as one
        band.
    reduction : str, list of str, or Reduction
        Windowing strategy.  Defaults to :class:`NonOverlappingWindows`, which
        matches the sequential-window preprocessing used to produce the reference
        embeddings on HuggingFace.
    """

    hf_repo: str = "light-curve/astromer2"
    hf_filename: str = "astromer2.onnx"
