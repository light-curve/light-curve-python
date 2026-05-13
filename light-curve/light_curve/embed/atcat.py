from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from light_curve.embed.input_tensors import InputTensors
from light_curve.embed.model import ExplicitMultiBandModel
from light_curve.embed.reduction import Reduction

if TYPE_CHECKING:
    from typing import Self


@dataclass
class ATCATInputs(InputTensors):
    flux: np.ndarray = field(kw_only=True)
    flux_err: np.ndarray = field(kw_only=True)
    time: np.ndarray = field(kw_only=True)
    mask: np.ndarray = field(kw_only=True)
    channel_index: np.ndarray = field(kw_only=True)


class ATCAT(ExplicitMultiBandModel):
    seq_size: int = 243
    valid_model_bands: frozenset[int] = frozenset(range(6))
    hf_repo: str = "light-curve/atcat"
    hf_filename: str = "atcat.onnx"
    model_outputs: frozenset[str] = frozenset({"last", "mean", "sequence"})

    def __init__(
        self,
        session,
        *,
        output: Literal["last", "mean", "sequence"] = "last",
        band_groups: Mapping | Sequence[Mapping] | None = None,
        allow_extra_bands: bool = False,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        reduction_kwargs: dict[str, object] | None = None,
        mag_zp: float = 31.4,
    ):
        super().__init__(
            session,
            band_groups=band_groups,
            allow_extra_bands=allow_extra_bands,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
        )
        if output not in self.model_outputs:
            raise ValueError(f"Unknown output '{output}'. Must be one of: {', '.join(sorted(self.model_outputs))}")
        self.output = output
        self.mag_zp = mag_zp

        fp16 = "float16" in session.get_inputs()[0].type
        self.dtype = np.float16 if fp16 else np.float32

        if abs(mag_zp - 27.5) < 1e-4:
            self.flux_scale = None
        else:
            self.flux_scale = 10 ** (-0.4 * (mag_zp - 27.5))

    @classmethod
    def from_hf(
        cls,
        output: str = "last",
        use_fp16: bool = False,
        *,
        band_groups: Sequence[str | int] | None = None,
        allow_extra_bands: bool = False,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        reduction_kwargs: dict[str, object] | None = None,
        mag_zp: float = 31.4,
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

            * ``"last"`` — embedding of the last valid timestep,
              output shape ``(band_groups, reductions, 1, 384)``
            * ``"mean"`` (default) — masked mean pooling over valid timesteps,
              output shape ``(band_groups, reductions, 1, 384)``
            * ``"sequence"`` — per-timestep embeddings (no aggregation),
              output shape ``(band_groups, reductions, 243, 384)``

        use_fp16 : bool, optional
            Whether to load the model in float16 precision if supported.
            Defaults to ``False`` (float32) for maximum compatibility; set to
            ``True`` to use original model precision and reduce memory usage
            if your hardware supports it.
        band_groups : sequence of str or int or None, optional
            Ordered band labels to embed.  ``None`` (default) treats the whole
            light curve as one band.
        allow_extra_bands: bool, optional
            If ``False`` (default), raises an error if the input light curve contains
            any bands which may not be used by the model: either bands specified as
            ``dict`` keys in ``band_groups`` or integers 0,1,2,3,4,5 (LSST ugrizy)
            if ``band_groups`` is ``None``.
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
            If ``output`` is not one of the recognized output names.
        ImportError
            If ``huggingface_hub`` is not installed.
        ImportError
            If no ``onnxruntime`` variant is installed.
        """
        filename = "atcat_fp16.onnx" if use_fp16 else "atcat_f32.onnx"
        return super().from_hf(
            filename,
            output=output,
            band_groups=band_groups,
            allow_extra_bands=allow_extra_bands,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
            mag_zp=mag_zp,
            ort_session_kwargs=ort_session_kwargs,
        )

    def __call__(self, time: ArrayLike, flux: ArrayLike, flux_err: ArrayLike, band: ArrayLike) -> np.ndarray:
        """Embed a light curve.

        Parameters
        ----------
        time: array-like, shape ``(n,)``
            Observation times in days (e.g. MJD).
        flux: array-like, shape ``(n,)``
            AB-calibrated bandflux, zero-point is given by ``self.mag_zp``.
        flux_err: array-like, shape ``(n,)``
            Uncertainties on the fluxes, in the same units as ``flux``.
        band: array-like, shape ``(n,)``
            Passband labeles, if ``self.band_groups`` is ``None``, integers 0-5
            corresponding to LSST ugrizy band_groups are expected. Otherwise, the
            keys are given by ``self.band_groups``.

        Returns
        -------
        np.ndarray, shape ``(n_bands, n_subsamples, seq_size, embed_dim)``
            Embedding tensor.  ``n_bands`` is 1 for ``self.band_groups`` is ``None`` or
            a single dict, and is equal to the number of dicts in ``band_groups`` for
            ``band_groups`` is a list of dicts.

        Raises
        ------
        ValueError
                If ``band`` contains values not in the defined input band_groups (when ``allow_extra_bands=False``).
        """
        return super().__call__(time, flux, flux_err, band=band)

    def preprocess_lc(self, time: ArrayLike, flux: ArrayLike, flux_err: ArrayLike, band: ArrayLike) -> ATCATInputs:
        """Preprocess a light curve into Astromer model input tensors.

        Parameters
        ----------
        time: array-like, shape ``(n,)``
            Observation times in days (e.g. MJD).
        flux: array-like, shape ``(n,)``
            AB-calibrated bandflux, zero-point is given by ``self.mag_zp``.
        flux_err: array-like, shape ``(n,)``
            Uncertainties on the fluxes, in the same units as ``flux``.
        band: array-like of int, shape ``(n,)``
            Passband labels, 0,1,2,3,4,5 (LSST ugrizy).

        Returns
        -------
        ATCATInputs
        """
        time, flux, flux_err, band, bool_mask = self.reduction.preprocess_lc(
            time, flux, flux_err, band, seq_size=self.seq_size
        )
        first_valid_idx = np.argmax(bool_mask, axis=-1, keepdims=True)
        first_time = np.take_along_axis(time, first_valid_idx, axis=-1)
        time = np.where(bool_mask, time - first_time, time)
        if self.flux_scale is not None:
            flux = np.where(bool_mask, flux * self.flux_scale, flux)
            flux_err = np.where(bool_mask, flux_err * self.flux_scale, flux_err)

        return ATCATInputs(
            bool_mask=bool_mask,
            flux=flux.astype(self.dtype),
            flux_err=flux_err.astype(self.dtype),
            time=time.astype(self.dtype),
            mask=bool_mask,
            channel_index=band.astype(np.int64),
        )

    def predict_tensors(self, tensors: ATCATInputs) -> np.ndarray:
        """Run the ONNX model on pre-processed tensors and return reduced embeddings.

        Parameters
        ----------
        tensors : ATCATInputs
            As returned by :meth:`preprocess_lc`.

        Returns
        -------
        np.ndarray, shape ``(n_subsamples, seq_size, embed_dim)``
            Embeddings after applying the time reduction's aggregation.
            For aggregated models (mean / last) ``seq_size`` is 1.
        """
        (raw_embedding,) = self.session.run([self.output], tensors.asdict())

        # Aggregated outputs (mean / max) have shape (n_windows, embed_dim); add SEQUENCE=1 axis
        if raw_embedding.ndim == 2:
            raw_embedding = np.expand_dims(raw_embedding, axis=1)

        return self.reduction.reduce_embeddings(raw_embedding, tensors, output=self.output)
