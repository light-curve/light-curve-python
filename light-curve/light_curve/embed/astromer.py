from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from .model import AstromerInputs, SingleBandModel
from .reduction import Reduction

_ONNX_INSTALL_HINT = (
    "An ONNX runtime is required to run embedding models. "
    "Install the variant that matches your hardware:\n"
    "  CPU / Apple Silicon:  pip install onnxruntime\n"
    "  NVIDIA GPU (CUDA):    pip install onnxruntime-gpu\n"
    "  Windows DirectML:     pip install onnxruntime-directml\n"
    "See https://onnxruntime.ai for the full list of packages."
)


def create_onnx_session(model_path: str, **kwargs):
    """Create an ``onnxruntime.InferenceSession``, with a helpful error if the package is missing.

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file.
    **kwargs
        Forwarded verbatim to ``onnxruntime.InferenceSession``.

    Returns
    -------
    onnxruntime.InferenceSession
        A ready-to-use inference session.

    Raises
    ------
    ImportError
        If no onnxruntime variant is installed, with installation instructions.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(_ONNX_INSTALL_HINT) from exc
    return ort.InferenceSession(model_path, **kwargs)


class _AstromerModel(SingleBandModel):
    """Internal base class for Astromer-family embedding models.

    Provides shared preprocessing (per-window zero-mean normalisation) and
    ONNX inference logic for all Astromer variants.  Concrete subclasses set
    :attr:`_HF_REPO` and, if needed, override the default ``reduction``.

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
    _HF_REPO: str
    _OUTPUTS: frozenset[str] = frozenset({"mean", "max", "sequence"})

    def __init__(
        self,
        session,
        *,
        output: str = "mean",
        bands: Sequence[str | int] | None = None,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        time_red_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            session,
            bands=bands,
            reduction=reduction,
            time_red_kwargs=time_red_kwargs,
        )
        if output not in self._OUTPUTS:
            raise ValueError(f"Unknown output '{output}'. Must be one of: {', '.join(sorted(self._OUTPUTS))}")
        self.output = output

    @classmethod
    def from_hf(
        cls,
        output: str = "mean",
        *,
        bands: Sequence[str | int] | None = None,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        time_red_kwargs: dict[str, object] | None = None,
        providers=None,
        sess_options=None,
    ) -> "_AstromerModel":
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
              output shape ``(batch, 256)``
            * ``"max"`` — masked max pooling over valid timesteps,
              output shape ``(batch, 256)``
            * ``"sequence"`` — per-timestep embeddings (no aggregation),
              output shape ``(batch, seq_size, 256)``

        bands : sequence of str or int, optional
            Ordered band labels to embed.  ``None`` (default) treats the whole
            light curve as one band.
        reduction : str, list of str, or Reduction, optional
            Windowing / subsampling strategy.  Defaults to
            ``"non-overlapping-windows"``.
        time_red_kwargs : dict, optional
            Extra keyword arguments forwarded to :func:`reduction_from_str`
            when ``reduction`` is given as a string.
        providers : list of str, optional
            ONNX Runtime execution providers, e.g.
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
        sess_options : onnxruntime.SessionOptions, optional
            Advanced session configuration passed directly to
            ``onnxruntime.InferenceSession``.

        Returns
        -------
        instance of the calling class
            Instance with a live ONNX inference session.

        Raises
        ------
        ValueError
            If ``output`` is not one of the recognised output names.
        ImportError
            If ``huggingface_hub`` is not installed, with instructions to
            install it or to download the model file manually.
        ImportError
            If no ``onnxruntime`` variant is installed, with hardware-specific
            installation instructions.
        """
        if output not in cls._OUTPUTS:
            raise ValueError(f"Unknown output '{output}'. Must be one of: {', '.join(sorted(cls._OUTPUTS))}")

        model_prefix = cls._HF_REPO.split("/")[-1]
        filename = f"{model_prefix}.onnx"

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            hf_url = f"https://huggingface.co/{cls._HF_REPO}/resolve/main/{filename}"
            raise ImportError(
                "huggingface_hub is required to download models from HuggingFace.\n"
                "Install it with:\n"
                "  pip install huggingface-hub\n"
                "Or download the model file directly:\n"
                f"  {hf_url}\n"
                "then load it with:\n"
                "  import onnxruntime as ort\n"
                f'  {cls.__name__}(session=ort.InferenceSession("/path/to/{filename}"), output="{output}")'
            ) from exc

        model_path = hf_hub_download(repo_id=cls._HF_REPO, filename=filename)

        session_kwargs = {}
        if providers is not None:
            session_kwargs["providers"] = providers
        if sess_options is not None:
            session_kwargs["sess_options"] = sess_options

        session = create_onnx_session(model_path, **session_kwargs)
        return cls(
            session=session,
            output=output,
            bands=bands,
            reduction=reduction,
            time_red_kwargs=time_red_kwargs,
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
        return AstromerInputs(times=time[idx], input=mag[idx], mask=mask[idx], bool_mask=bool_mask)

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
            {"input": tensors.input, "times": tensors.times, "mask_in": tensors.mask},
        )

        # Aggregated outputs (mean / max) have shape (n_windows, embed_dim); add SEQUENCE=1 axis
        if raw_embedding.ndim == 2:
            raw_embedding = np.expand_dims(raw_embedding, axis=1)

        return self.reduction.reduce_embeddings(raw_embedding, tensors, output=self.output)


class Astromer1(_AstromerModel):
    """Astromer 1 embedding model (Donoso-Oliva et al. 2023).

    Transformer encoder pretrained on MACHO R-band light curves via masked
    magnitude prediction.  Accepts irregularly-sampled single-band photometry
    and returns a 256-dimensional embedding (2 layers, 4 attention heads).

    The ONNX model is hosted on HuggingFace at
    ``https://huggingface.co/light-curve/astromer1`` (``astromer1.onnx``).
    Three named outputs are available; select with the ``output`` parameter:

    * ``"mean"`` (default) — masked mean pooling → shape ``(batch, 256)``
    * ``"max"`` — masked max pooling → shape ``(batch, 256)``
    * ``"sequence"`` — per-timestep features → shape ``(batch, 200, 256)``

    Use :meth:`from_hf` to download and load the model directly.

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

    _HF_REPO = "light-curve/astromer1"


class Astromer2(_AstromerModel):
    """Astromer 2 embedding model (Donoso-Oliva et al. 2026).

    Pretrained on 1.5 million MACHO light curves.  Accepts irregularly-sampled
    single-band photometry and returns a 256-dimensional embedding.

    The ONNX model is hosted on HuggingFace at
    ``https://huggingface.co/light-curve/astromer2`` (``astromer2.onnx``).
    Three named outputs are available; select with the ``output`` parameter:

    * ``"mean"`` (default) — masked mean pooling → shape ``(batch, 256)``
    * ``"max"`` — masked max pooling → shape ``(batch, 256)``
    * ``"sequence"`` — per-timestep features → shape ``(batch, 200, 256)``

    Use :meth:`from_hf` to download and load the model directly.

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

    _HF_REPO = "light-curve/astromer2"
