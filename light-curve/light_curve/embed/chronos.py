from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import ArrayLike

from light_curve.embed.input_tensors import InputTensors
from light_curve.embed.model import (
    SingleBandModel,
    _hf_hub_download_cached,
    create_onnx_session,
)
from light_curve.embed.reduction import Reduction

if TYPE_CHECKING:
    from typing import Self

    import onnxruntime as ort

_PATCH_SIZE = 16


@dataclass
class ChronosInputs(InputTensors):
    """Input tensors for Chronos-family models.

    Attributes
    ----------
    mag : ndarray, shape ``(n_subsamples, seq_size)``
        Per-subsample magnitudes, zero-padded to the reduction's ``seq_size``.
        The actual model context (left NaN-padded to a multiple of the patch
        size) is built per subsample at inference time from the valid entries.
    bool_mask : ndarray, shape ``(n_subsamples, seq_size)``
        Boolean validity — ``True`` for real observations, ``False`` for padding.
    """

    mag: np.ndarray = field(kw_only=True)


class _ChronosModel(SingleBandModel):
    """Internal base class for Chronos-family embedding models.

    Chronos models embed a single univariate magnitude series: timestamps are
    discarded and observations are treated as sequentially ordered (the
    StarEmbed approach).  The series is left-padded with NaN to a multiple of
    the patch size (16); instance normalisation (mean subtraction, std scaling,
    arcsinh) is applied internally by the model.

    Concrete subclasses set :attr:`max_obs`, :attr:`embed_dim`, and the
    HuggingFace location.

    Output shape
    ------------
    :meth:`__call__` returns ``(1, n_subsamples, seq_size, embed_dim)``:

    * the leading band axis is always 1 (Chronos is single-series);
    * ``n_subsamples`` is the number of reduction windows (1 for the default
      ``"end"`` reduction);
    * ``seq_size`` is 1 for the ``"mean"`` output and the number of patches for
      the ``"sequence"`` output.
    """

    patch_size: int = _PATCH_SIZE
    max_obs: int
    embed_dim: int
    model_outputs: frozenset[str] = frozenset({"mean", "sequence"})

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        output: Literal["mean", "sequence"] = "mean",
        reduction: str | list[str] | Reduction = "end",
        reduction_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            session,
            bands=None,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
        )
        if output not in self.model_outputs:
            raise ValueError(f"Unknown output '{output}'. Must be one of: {', '.join(sorted(self.model_outputs))}")
        self.output = output

    @classmethod
    def from_hf(
        cls,
        output: str = "mean",
        *,
        reduction: str | list[str] | Reduction = "end",
        reduction_kwargs: dict[str, object] | None = None,
        ort_session_kwargs: dict[str, object] | None = None,
    ) -> Self:
        """Load a model from the HuggingFace Hub.

        Downloads (and caches) the ONNX model file, creates an
        ``onnxruntime.InferenceSession``, and returns a ready-to-use instance.

        Parameters
        ----------
        output : str, optional
            Named ONNX output to return: ``"mean"`` (default, masked mean pool
            over valid context patches → ``(..., 1, embed_dim)``) or
            ``"sequence"`` (per-patch encoder states → ``(..., n_patches,
            embed_dim)``).
        reduction : str, list of str, or Reduction, optional
            Strategy for selecting observations when a light curve exceeds
            ``max_obs``.  Defaults to ``"end"`` (the most recent ``max_obs``
            observations, matching the model's native right-aligned context).
        reduction_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :func:`reduction_from_str`.
        ort_session_kwargs : dict or None, optional
            Keyword arguments forwarded to ``onnxruntime.InferenceSession``.

        Returns
        -------
        instance of the calling class
            Instance with a live ONNX inference session.

        Raises
        ------
        ValueError
            If ``output`` is not one of the recognised output names.
        ImportError
            If ``huggingface_hub`` or an ``onnxruntime`` variant is missing.
        """
        return super().from_hf(
            cls.hf_filename,
            output=output,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
            ort_session_kwargs=ort_session_kwargs,
        )

    def __call__(self, mag: ArrayLike) -> np.ndarray:
        """Embed a magnitude series.

        Parameters
        ----------
        mag : array-like, shape ``(n,)``
            Magnitudes in chronological order.  Timestamps are not used by the
            model, which treats observations as sequentially ordered.

        Returns
        -------
        np.ndarray, shape ``(1, n_subsamples, seq_size, embed_dim)``
            Embedding tensor.  ``seq_size`` is 1 for ``"mean"`` and the number
            of patches for ``"sequence"``.
        """
        return super().__call__(mag)

    def preprocess_lc(self, mag: ArrayLike) -> ChronosInputs:
        """Select observations per the reduction; padding to the patch multiple is deferred.

        Parameters
        ----------
        mag : array-like, shape ``(n,)``
            Magnitudes in chronological order.

        Returns
        -------
        ChronosInputs
        """
        mag = np.asarray(mag, dtype=np.float32)
        mag_win, bool_mask = self.reduction.preprocess_lc(mag, seq_size=self.max_obs)
        return ChronosInputs(bool_mask=bool_mask, mag=mag_win.astype(np.float32))

    def _context(self, mag: np.ndarray) -> np.ndarray:
        """Left-pad valid magnitudes with NaN to the next multiple of the patch size."""
        n = mag.shape[0]
        seq = max(self.patch_size, -(-n // self.patch_size) * self.patch_size)
        context = np.full((1, seq), np.nan, dtype=np.float32)
        context[0, seq - n :] = mag
        return context

    def predict_tensors(self, tensors: ChronosInputs) -> np.ndarray:
        """Run the ONNX model per subsample and return reduced embeddings.

        Parameters
        ----------
        tensors : ChronosInputs
            As returned by :meth:`preprocess_lc`.

        Returns
        -------
        np.ndarray, shape ``(n_subsamples, seq_size, embed_dim)``
            Embeddings after applying the reduction's aggregation.  ``seq_size``
            is 1 for ``"mean"``.

        Raises
        ------
        ValueError
            For the ``"sequence"`` output with a multi-window reduction, since
            per-window patch counts differ and cannot be stacked.
        """
        outputs = []
        for subsample in range(tensors.bool_mask.shape[0]):
            valid = tensors.bool_mask[subsample]
            context = self._context(tensors.mag[subsample][valid])
            (raw,) = self.session.run([self.output], {"context": context})
            outputs.append(raw[0])  # mean: (embed_dim,); sequence: (n_patches, embed_dim)

        if self.output == "mean":
            embeddings = np.stack(outputs)[:, np.newaxis, :]  # (n_subsamples, 1, embed_dim)
            return self.reduction.reduce_embeddings(embeddings, tensors, output=self.output)

        # "sequence": per-window patch counts vary, so only a single window is supported.
        if len(outputs) != 1:
            raise ValueError(
                "The 'sequence' output supports only single-subsample reductions for Chronos "
                "(per-window patch counts differ and cannot be stacked)."
            )
        embeddings = outputs[0][np.newaxis, ...]  # (1, n_patches, embed_dim)
        return self.reduction.reduce_embeddings(embeddings, tensors, output=self.output)


class Chronos2(_ChronosModel):
    """Chronos 2 univariate light-curve embedding model.

    A T5-style transformer encoder pretrained on a large corpus of real and
    synthetic time series.  It maps a magnitude sequence to 768-dimensional
    patch embeddings (native context up to 8192 observations).

    The ONNX model is hosted on HuggingFace at
    ``https://huggingface.co/light-curve/chronos2`` (``chronos2.onnx``).

    Use :meth:`from_hf` to download and load the model directly.

    Model license
    -------------
    Apache-2.0 (upstream amazon/chronos-2 license).

    References
    ----------
    Ansari et al. (2024), *Chronos: Learning the Language of Time Series*,
    Transactions on Machine Learning Research.
    https://huggingface.co/amazon/chronos-2

    Parameters
    ----------
    session :
        ONNX inference session for the Chronos 2 model file.
    output : str, optional
        ``"mean"`` (default) or ``"sequence"``.
    reduction : str, list of str, or Reduction, optional
        Observation-selection strategy for light curves longer than 8192.
        Defaults to ``"end"``.
    reduction_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`reduction_from_str`.
    """

    max_obs: int = 8192
    embed_dim: int = 768
    hf_repo: str = "light-curve/chronos2"
    hf_filename: str = "chronos2.onnx"


class ChronosBolt(_ChronosModel):
    """Chronos-Bolt univariate light-curve embedding model.

    A faster, patch-based Chronos variant available in four sizes with
    different embedding dimensions (native context up to 2048 observations):

    ===== =========
    size  embed_dim
    ===== =========
    tiny  256
    mini  384
    small 512
    base  768
    ===== =========

    The ONNX models are hosted on HuggingFace at
    ``https://huggingface.co/light-curve/chronos-bolt-<size>``.

    Use :meth:`from_hf` (with ``size=``) to download and load the model.

    Model license
    -------------
    Apache-2.0 (upstream amazon/chronos-bolt license).

    References
    ----------
    Ansari et al. (2024), *Chronos: Learning the Language of Time Series*,
    Transactions on Machine Learning Research.
    https://huggingface.co/amazon/chronos-bolt-base

    Parameters
    ----------
    session :
        ONNX inference session for the Chronos-Bolt model file.
    size : {"tiny", "mini", "small", "base"}
        Which model size this session corresponds to (sets ``embed_dim``).
    output : str, optional
        ``"mean"`` (default) or ``"sequence"``.
    reduction : str, list of str, or Reduction, optional
        Observation-selection strategy for light curves longer than 2048.
        Defaults to ``"end"``.
    reduction_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`reduction_from_str`.
    """

    max_obs: int = 2048
    _EMBED_DIMS: dict[str, int] = {"tiny": 256, "mini": 384, "small": 512, "base": 768}

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        size: Literal["tiny", "mini", "small", "base"],
        output: Literal["mean", "sequence"] = "mean",
        reduction: str | list[str] | Reduction = "end",
        reduction_kwargs: dict[str, object] | None = None,
    ) -> None:
        if size not in self._EMBED_DIMS:
            raise ValueError(f"Unknown size '{size}'. Must be one of: {', '.join(sorted(self._EMBED_DIMS))}")
        self.size = size
        self.embed_dim = self._EMBED_DIMS[size]
        self.hf_repo = f"light-curve/chronos-bolt-{size}"
        self.hf_filename = f"chronos-bolt-{size}.onnx"
        super().__init__(
            session,
            output=output,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
        )

    @classmethod
    def from_hf(
        cls,
        size: str = "base",
        output: str = "mean",
        *,
        reduction: str | list[str] | Reduction = "end",
        reduction_kwargs: dict[str, object] | None = None,
        ort_session_kwargs: dict[str, object] | None = None,
    ) -> Self:
        """Load a Chronos-Bolt model of the given ``size`` from the HuggingFace Hub.

        Parameters
        ----------
        size : {"tiny", "mini", "small", "base"}, optional
            Model size to load.  Defaults to ``"base"``.
        output : str, optional
            ``"mean"`` (default) or ``"sequence"``.
        reduction : str, list of str, or Reduction, optional
            Observation-selection strategy for light curves longer than 2048.
            Defaults to ``"end"``.
        reduction_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :func:`reduction_from_str`.
        ort_session_kwargs : dict or None, optional
            Keyword arguments forwarded to ``onnxruntime.InferenceSession``.

        Returns
        -------
        ChronosBolt
            Instance with a live ONNX inference session.

        Raises
        ------
        ValueError
            If ``size`` or ``output`` is not recognised.
        ImportError
            If ``huggingface_hub`` or an ``onnxruntime`` variant is missing.
        """
        if size not in cls._EMBED_DIMS:
            raise ValueError(f"Unknown size '{size}'. Must be one of: {', '.join(sorted(cls._EMBED_DIMS))}")
        model_path = _hf_hub_download_cached(f"light-curve/chronos-bolt-{size}", f"chronos-bolt-{size}.onnx")
        session = create_onnx_session(model_path, **(ort_session_kwargs or {}))
        return cls(
            session=session,
            size=size,
            output=output,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
        )
