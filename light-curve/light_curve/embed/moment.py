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

# MOMENT has a fixed 512-step context split into 64 non-overlapping patches of 8.
_SEQ_LEN = 512
_PATCH_SIZE = 8


@dataclass
class MomentInputs(InputTensors):
    """Input tensors for MOMENT-1 models.

    Attributes
    ----------
    mag : ndarray, shape ``(n_subsamples, seq_size)``
        Per-subsample magnitudes, zero-padded to the reduction's ``seq_size``.
        The actual model context (left NaN-padded to the fixed 512-step window)
        is built per subsample at inference time from the valid entries.
    bool_mask : ndarray, shape ``(n_subsamples, seq_size)``
        Boolean validity — ``True`` for real observations, ``False`` for padding.
    """

    mag: np.ndarray = field(kw_only=True)


class Moment1(SingleBandModel):
    """MOMENT-1 univariate light-curve embedding model.

    A T5-based time-series foundation model (Goswami et al. 2024) pretrained with
    a masked-reconstruction objective on the Time-series Pile.  It embeds a single
    univariate magnitude series: timestamps are discarded and observations are
    treated as sequentially ordered (the same convention used for the Chronos
    models).  The series is capped to the most recent 512 observations and
    left-padded with NaN to that fixed window; reversible instance normalisation
    (RevIN) is applied internally by the model.

    The model comes in three sizes with different embedding dimensions: ``small``
    (512), ``base`` (768), and ``large`` (1024).  Unlike Chronos, the context
    length is fixed at 512 observations (64 patches of 8), not a dynamic axis.

    The ONNX models are hosted on HuggingFace at
    ``https://huggingface.co/light-curve/moment1-<size>``.

    Use :meth:`from_hf` (with ``size=``) to download and load the model.

    Model license
    -------------
    MIT (upstream AutonLab/MOMENT-1 license).

    References
    ----------
    Goswami et al. (2024), *MOMENT: A Family of Open Time-series Foundation
    Models*, ICML 2024.  https://huggingface.co/AutonLab/MOMENT-1-base

    Parameters
    ----------
    session :
        ONNX inference session for the MOMENT-1 model file.
    size : {"small", "base", "large"}
        Which model size this session corresponds to (sets ``embed_dim``).
    output : str, optional
        ``"mean"`` (default) or ``"sequence"``.
    reduction : str, list of str, or Reduction, optional
        Observation-selection strategy for light curves longer than 512.
        Defaults to ``"end"``.
    reduction_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`reduction_from_str`.
    """

    patch_size: int = _PATCH_SIZE
    seq_len: int = _SEQ_LEN
    max_obs: int = _SEQ_LEN
    model_outputs: frozenset[str] = frozenset({"mean", "sequence"})
    _EMBED_DIMS: dict[str, int] = {"small": 512, "base": 768, "large": 1024}

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        size: Literal["small", "base", "large"],
        output: Literal["mean", "sequence"] = "mean",
        reduction: str | list[str] | Reduction = "end",
        reduction_kwargs: dict[str, object] | None = None,
    ) -> None:
        if size not in self._EMBED_DIMS:
            raise ValueError(f"Unknown size '{size}'. Must be one of: {', '.join(sorted(self._EMBED_DIMS))}")
        self.size = size
        self.embed_dim = self._EMBED_DIMS[size]
        self.hf_repo = f"light-curve/moment1-{size}"
        self.hf_filename = f"moment1-{size}.onnx"
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
        size: str,
        output: str = "mean",
        *,
        reduction: str | list[str] | Reduction = "end",
        reduction_kwargs: dict[str, object] | None = None,
        ort_session_kwargs: dict[str, object] | None = None,
    ) -> Self:
        """Load a MOMENT-1 model of the given ``size`` from the HuggingFace Hub.

        Downloads (and caches) the ONNX model file, creates an
        ``onnxruntime.InferenceSession``, and returns a ready-to-use instance.

        Parameters
        ----------
        size : {"small", "base", "large"}
            Model size to load.  Required: the sizes have different embedding
            dimensions, so there is no meaningful default.
        output : str, optional
            Named ONNX output to return: ``"mean"`` (default, masked mean pool
            over valid patches → ``(..., 1, embed_dim)``) or ``"sequence"``
            (per-patch encoder states → ``(..., 64, embed_dim)``).
        reduction : str, list of str, or Reduction, optional
            Observation-selection strategy for light curves longer than 512.
            Defaults to ``"end"`` (the most recent 512 observations, matching the
            model's native right-aligned context).
        reduction_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :func:`reduction_from_str`.
        ort_session_kwargs : dict or None, optional
            Keyword arguments forwarded to ``onnxruntime.InferenceSession``.

        Returns
        -------
        Moment1
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
        model_path = _hf_hub_download_cached(f"light-curve/moment1-{size}", f"moment1-{size}.onnx")
        session = create_onnx_session(model_path, **(ort_session_kwargs or {}))
        return cls(
            session=session,
            size=size,
            output=output,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
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
            Embedding tensor.  ``seq_size`` is 1 for ``"mean"`` and 64 (the
            number of patches) for ``"sequence"``.
        """
        return super().__call__(mag)

    def preprocess_lc(self, mag: ArrayLike) -> MomentInputs:
        """Select observations per the reduction; padding to the fixed window is deferred.

        Parameters
        ----------
        mag : array-like, shape ``(n,)``
            Magnitudes in chronological order.

        Returns
        -------
        MomentInputs
        """
        mag = np.asarray(mag, dtype=np.float32)
        mag_win, bool_mask = self.reduction.preprocess_lc(mag, seq_size=self.max_obs)
        return MomentInputs(bool_mask=bool_mask, mag=mag_win.astype(np.float32))

    def _context(self, mag: np.ndarray) -> np.ndarray:
        """Left-pad valid magnitudes with NaN to the fixed 512-step window."""
        mag = mag[-self.seq_len :]
        n = mag.shape[0]
        context = np.full((1, self.seq_len), np.nan, dtype=np.float32)
        context[0, self.seq_len - n :] = mag
        return context

    def predict_tensors(self, tensors: MomentInputs) -> np.ndarray:
        """Run the ONNX model per subsample and return reduced embeddings.

        Because MOMENT's context length is fixed (512), all subsamples share the
        same shape and are batched into a single ONNX call.

        Parameters
        ----------
        tensors : MomentInputs
            As returned by :meth:`preprocess_lc`.

        Returns
        -------
        np.ndarray, shape ``(n_subsamples, seq_size, embed_dim)``
            Embeddings after applying the reduction's aggregation.  ``seq_size``
            is 1 for ``"mean"`` and 64 for ``"sequence"``.

        Raises
        ------
        ValueError
            For the ``"sequence"`` output with a multi-window reduction: the
            reduction's per-window aggregation operates in observation space,
            which does not align with the fixed 64-patch sequence.
        """
        n_subsamples = tensors.bool_mask.shape[0]
        if self.output == "sequence" and n_subsamples != 1:
            raise ValueError(
                "The 'sequence' output supports only single-subsample reductions for MOMENT "
                "(per-window aggregation operates in observation space, which does not align "
                "with the fixed 64-patch sequence)."
            )

        contexts = np.concatenate(
            [self._context(tensors.mag[i][tensors.bool_mask[i]]) for i in range(n_subsamples)],
            axis=0,
        )  # (n_subsamples, 512)
        (raw,) = self.session.run([self.output], {"context": contexts})
        # mean: (n_subsamples, embed_dim); sequence: (n_subsamples, 64, embed_dim)

        if self.output == "mean":
            embeddings = raw[:, np.newaxis, :]  # (n_subsamples, 1, embed_dim)
        else:
            embeddings = raw  # (1, 64, embed_dim)
        return self.reduction.reduce_embeddings(embeddings, tensors, output=self.output)
