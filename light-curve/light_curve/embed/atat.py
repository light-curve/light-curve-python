from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from light_curve.embed.input_tensors import InputTensors
from light_curve.embed.model import MultiBandModel
from light_curve.embed.reduction import Reduction

if TYPE_CHECKING:
    from typing import Self

    import onnxruntime as ort


@dataclass
class ATATInputs(InputTensors):
    """Input tensors for the ATAT embedding model.

    All arrays share the subsample axis ``S`` (one slice per reduction window),
    a fixed per-band sequence length of 65, and the 6-band axis (LSST *u g r i
    z Y*).

    Attributes
    ----------
    data : ndarray, shape ``(S, 65, 6)``
        Per-band flux (SNANA FLUXCAL, ZP = 27.5); padding slots hold ``0.0``.
    time : ndarray, shape ``(S, 65, 6)``
        Per-band observation times in days, shifted so the earliest valid
        observation (across all bands) is ``0``; padding slots hold ``0.0``.
    mask : ndarray, shape ``(S, 65, 6)``
        Validity mask — ``1.0`` for real observations, ``0.0`` for padding.
    bool_mask : ndarray, shape ``(S, 65 * 6)``
        Boolean validity flattened over the (sequence, band) axes — ``True``
        for real observations, ``False`` for padding.
    """

    data: np.ndarray = field(kw_only=True)
    time: np.ndarray = field(kw_only=True)
    mask: np.ndarray = field(kw_only=True)


# SNANA FLUXCAL reference zero-point: a source at 27.5 AB mag has FLUXCAL = 1.
_ATAT_ZP = 27.5


class ATAT(MultiBandModel):
    """ATAT multiband transformer embedding model.

    ATAT (Astronomical Transformer for time series And Tabular data) is a
    transformer encoder for irregularly-sampled, multi-band light curves.  Each
    of the six photometric bands is embedded independently with a learned
    sinusoidal time modulation, the bands are then merged, sorted by observation
    time, and passed through a multi-head self-attention transformer with a
    learnable CLS token.  The CLS-token output is the default representation
    used in the paper.  ATAT was trained for transient classification on the
    ELAsTiCC simulation (20 classes, LSST-like photometry).

    The model expects raw fluxes calibrated to AB zero-point 27.5 (ELAsTiCC /
    SNANA FITS convention), with **no** normalisation.  Use ``mag_zp`` to
    convert from a different zero-point at call time — common values are 31.4
    (LSST nJy) and 8.9 (Jy).

    Valid model band indices are 0–5, corresponding to LSST *u g r i z Y*.
    Pass a ``band_groups`` dict (e.g. ``{"u": 0, "g": 1, ...}``) to use string
    band labels instead of integers.

    Parameters
    ----------
    session :
        An ``onnxruntime.InferenceSession`` for the ATAT ONNX model.
    output : {"token", "mean", "sequence"}, optional
        Which model head to return:

        * ``"token"`` (default) — CLS-token embedding,
          output shape ``(n_band_groups, n_subsamples, 1, 192)``
        * ``"mean"`` — masked mean pooling over valid observations,
          output shape ``(n_band_groups, n_subsamples, 1, 192)``
        * ``"sequence"`` — per-observation embeddings (CLS token excluded),
          output shape ``(n_band_groups, n_subsamples, 390, 192)``

    band_groups : Mapping, list of Mapping, or None, optional
        Band label → model integer mapping(s).  See
        :class:`~light_curve.embed.model.MultiBandModel` for details.
    allow_extra_bands : bool, optional
        If ``False`` (default), raises :exc:`ValueError` when the input
        contains band labels not in ``band_groups`` (or not in 0–5 when
        ``band_groups`` is ``None``).
    reduction : str, list of str, or Reduction, optional
        Strategy for selecting up to 65 observations per band before
        zero-padding.  Defaults to ``"non-overlapping-windows"``.  Any
        :class:`~light_curve.embed.SingleSubsampleReduction` or
        :class:`~light_curve.embed.MultipleReductions` thereof works well;
        :class:`~light_curve.embed.NonOverlappingWindows` requires every band to
        produce the same number of windows.
    reduction_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`reduction_from_str`.
    mag_zp : float, optional
        AB zero-point of the input fluxes.  Fluxes are rescaled to ZP = 27.5
        (ELAsTiCC / SNANA FITS convention) before inference.  Common values:
        31.4 (LSST nJy, default), 27.5 (no rescaling needed), 8.9 (Jy).

    Examples
    --------
    >>> import numpy as np
    >>> from light_curve.embed import ATAT
    >>> model = ATAT.from_hf(
    ...     output="token",
    ...     band_groups={"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Y": 5},
    ... )
    >>> time = np.linspace(0, 200, 100, dtype=np.float32)
    >>> flux = np.ones(100, dtype=np.float32)
    >>> band = np.array(["g", "r"] * 50)
    >>> embedding = model(time, flux, band)
    >>> embedding.shape
    (1, 1, 1, 192)

    Model license
    -------------
    Apache-2.0 (upstream ATAT license).

    References
    ----------
    Becker et al. (2024), *ATAT: Astronomical Transformer for time series And
    Tabular data*, Astronomy & Astrophysics, 691, A163.
    https://doi.org/10.1051/0004-6361/202451418
    """

    seq_size: int = 65
    n_model_bands: int = 6
    embed_dim: int = 192
    hf_repo: str = "light-curve/atat"
    hf_filename: str = "atat.onnx"
    model_outputs: frozenset[str] = frozenset({"token", "mean", "sequence"})

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        output: Literal["token", "mean", "sequence"] = "token",
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

        if abs(mag_zp - _ATAT_ZP) < 1e-4:
            self.flux_scale = None
        else:
            self.flux_scale = 10 ** (-0.4 * (mag_zp - _ATAT_ZP))

    @classmethod
    def from_hf(
        cls,
        output: str = "token",
        *,
        band_groups: Mapping | Sequence[Mapping] | None = None,
        allow_extra_bands: bool = False,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        reduction_kwargs: dict[str, object] | None = None,
        mag_zp: float = 31.4,
        ort_session_kwargs: dict[str, object] | None = None,
    ) -> Self:
        """Load a model from the HuggingFace Hub.

        Downloads (and caches) the ONNX model file, creates an
        ``onnxruntime.InferenceSession``, and returns a ready-to-use instance.

        Parameters
        ----------
        output : str, optional
            Named ONNX output to return.  One of:

            * ``"token"`` (default) — CLS-token embedding,
              output shape ``(band_groups, reductions, 1, 192)``
            * ``"mean"`` — masked mean pooling over valid observations,
              output shape ``(band_groups, reductions, 1, 192)``
            * ``"sequence"`` — per-observation embeddings (CLS token excluded),
              output shape ``(band_groups, reductions, 390, 192)``

        band_groups : Mapping, list of Mapping, or None, optional
            Band label → model integer mapping.  ``None`` (default) expects
            integer band indices 0–5 (LSST *u g r i z Y*).
        allow_extra_bands : bool, optional
            If ``False`` (default), raises an error if the input light curve
            contains bands not used by the model: either ``dict`` keys in
            ``band_groups`` or integers 0–5 (LSST ugrizy) when ``band_groups``
            is ``None``.
        reduction : str, list of str, or Reduction, optional
            Per-band windowing / subsampling strategy.  Defaults to
            ``"non-overlapping-windows"``.
        reduction_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :func:`reduction_from_str`
            when ``reduction`` is given as a string.
        mag_zp : float, optional
            AB zero-point of the input fluxes.  Fluxes are rescaled to ZP = 27.5
            (ELAsTiCC / SNANA FITS convention) before inference.  Common values:
            31.4 (LSST nJy, default), 27.5 (no rescaling needed), 8.9 (Jy).
        ort_session_kwargs : dict or None, optional
            Additional keyword arguments forwarded to
            ``onnxruntime.InferenceSession``: "sess_options", "providers",
            "provider_options".

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
        return super().from_hf(
            cls.hf_filename,
            output=output,
            band_groups=band_groups,
            allow_extra_bands=allow_extra_bands,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
            mag_zp=mag_zp,
            ort_session_kwargs=ort_session_kwargs,
        )

    def __call__(self, time: ArrayLike, flux: ArrayLike, band: ArrayLike) -> np.ndarray:
        """Embed a light curve.

        Parameters
        ----------
        time : array-like, shape ``(n,)``
            Observation times in days (e.g. MJD).
        flux : array-like, shape ``(n,)``
            AB-calibrated bandflux; zero-point is given by ``self.mag_zp``.
        band : array-like, shape ``(n,)``
            Passband labels.  If ``self.band_groups`` is ``None``, integers 0–5
            corresponding to LSST ugrizy are expected.  Otherwise, the keys are
            given by ``self.band_groups``.

        Returns
        -------
        np.ndarray, shape ``(n_band_groups, n_subsamples, seq_size, embed_dim)``
            Embedding tensor.  ``n_band_groups`` is 1 when ``band_groups`` is
            ``None`` or a single dict, and equals the number of dicts when
            ``band_groups`` is a list of dicts.  ``seq_size`` is 1 for the
            ``"token"`` / ``"mean"`` outputs and 390 for ``"sequence"``.

        Raises
        ------
        ValueError
            If ``band`` contains values not in the defined input band groups
            (when ``allow_extra_bands=False``).
        """
        return super().__call__(time, flux, band=band)

    def preprocess_lc(self, time: ArrayLike, flux: ArrayLike, band: ArrayLike) -> ATATInputs:
        """Preprocess a light curve into ATAT model input tensors.

        Splits observations into the six model bands, applies the reduction to
        each band independently (selecting up to 65 observations and
        zero-padding), stacks the bands into the ``(S, 65, 6)`` layout, shifts
        time so the earliest valid observation is zero, and rescales flux to
        ZP = 27.5 when needed.

        Parameters
        ----------
        time : array-like, shape ``(n,)``
            Observation times in days (e.g. MJD).
        flux : array-like, shape ``(n,)``
            AB-calibrated bandflux; zero-point is given by ``self.mag_zp``.
        band : array-like of int, shape ``(n,)``
            Passband labels, 0–5 (LSST ugrizy).

        Returns
        -------
        ATATInputs
        """
        time = np.asarray(time)
        flux = np.asarray(flux)
        band = np.asarray(band)

        per_band_time = []
        per_band_flux = []
        per_band_mask = []
        n_subsamples: int | None = None
        for band_idx in range(self.n_model_bands):
            band_mask = band == band_idx
            time_win, flux_win, bool_mask_win = self.reduction.preprocess_lc(
                time[band_mask], flux[band_mask], seq_size=self.seq_size
            )
            if n_subsamples is None:
                n_subsamples = bool_mask_win.shape[0]
            elif bool_mask_win.shape[0] != n_subsamples:
                raise ValueError(
                    f"Band {band_idx} produced {bool_mask_win.shape[0]} subsamples "
                    f"but previous bands produced {n_subsamples}."
                )
            per_band_time.append(time_win)
            per_band_flux.append(flux_win)
            per_band_mask.append(bool_mask_win)

        # Stack along a trailing band axis: (n_subsamples, seq_size, n_bands).
        time_mat = np.stack(per_band_time, axis=-1)
        flux_mat = np.stack(per_band_flux, axis=-1)
        bool_mask = np.stack(per_band_mask, axis=-1)

        # Shift time so the earliest valid observation (across all bands) is 0.
        masked_time = np.where(bool_mask, time_mat, np.inf)
        t_min = masked_time.reshape(time_mat.shape[0], -1).min(axis=1)
        t_min = np.where(np.isfinite(t_min), t_min, 0.0)
        time_mat = np.where(bool_mask, time_mat - t_min[:, np.newaxis, np.newaxis], 0.0)

        flux_mat = np.where(bool_mask, flux_mat, 0.0)
        if self.flux_scale is not None:
            flux_mat = np.where(bool_mask, flux_mat * self.flux_scale, 0.0)

        return ATATInputs(
            bool_mask=bool_mask.reshape(bool_mask.shape[0], -1),
            data=flux_mat.astype(self.dtype),
            time=time_mat.astype(self.dtype),
            mask=bool_mask.astype(self.dtype),
        )

    def predict_tensors(self, tensors: ATATInputs) -> np.ndarray:
        """Run the ONNX model on pre-processed tensors and return reduced embeddings.

        Parameters
        ----------
        tensors : ATATInputs
            As returned by :meth:`preprocess_lc`.

        Returns
        -------
        np.ndarray, shape ``(n_subsamples, seq_size, embed_dim)``
            Embeddings after applying the time reduction's aggregation.
            For aggregated outputs (``token`` / ``mean``) ``seq_size`` is 1.
        """
        (raw_embedding,) = self.session.run([self.output], tensors.asdict())

        # Aggregated outputs (token / mean) have shape (n_subsamples, embed_dim);
        # add the SEQUENCE=1 axis to match the 3-D contract.
        if raw_embedding.ndim == 2:
            raw_embedding = np.expand_dims(raw_embedding, axis=1)

        return self.reduction.reduce_embeddings(raw_embedding, tensors, output=self.output)
