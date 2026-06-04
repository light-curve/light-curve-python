from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from light_curve.embed.input_tensors import InputTensors
from light_curve.embed.model import ImplicitMultiBandModel
from light_curve.embed.reduction import Reduction

if TYPE_CHECKING:
    from typing import Self

    import onnxruntime as ort


@dataclass
class AstraCLRInputs(InputTensors):
    """Input tensors for the AstraCLR embedding model.

    All arrays share the subsample axis ``S`` (one slice per reduction window)
    and the fixed concatenated sequence length 700 (300 g + 350 r + 50 i).

    Attributes
    ----------
    input : ndarray, shape ``(S, 700, 1)``
        Inverse-variance weighted mean-subtracted magnitudes (per-band).
    times : ndarray, shape ``(S, 700, 1)``
        MJD observation times shifted by :data:`_MJD_OFFSET` (58 000).
    band_info : ndarray, shape ``(S, 700, 1)``
        log10 effective wavelength (Å) for each observation's ZTF band.
    mask : ndarray, shape ``(S, 700)``
        Padding mask — ``1.0`` for padded positions, ``0.0`` for real observations.
    bool_mask : ndarray, shape ``(S, 700)``
        Boolean validity — ``True`` for real, ``False`` for padded.
    """

    input: np.ndarray = field(kw_only=True)
    times: np.ndarray = field(kw_only=True)
    band_info: np.ndarray = field(kw_only=True)
    mask: np.ndarray = field(kw_only=True)


_SEQUENCE_LENGTH = 700
_SEQ_PER_BAND = [300, 350, 50]  # g, r, i
_MJD_OFFSET = 58_000.0
_LG_EFF_WAVE = [np.log10(4746.48), np.log10(6366.38), np.log10(7829.03)]  # g, r, i


class AstraCLR(ImplicitMultiBandModel):
    """AstraCLR multi-band ZTF embedding model.

    AstraCLR is a transformer
    encoder pretrained on ZTF photometry via a contrastive learning objective.
    It accepts magnitudes in the ZTF *g*, *r*, and *i* bands and returns a
    single 512-dimensional embedding per light curve.

    Observations are split by band (integers 0, 1, 2 for *g*, *r*, *i*),
    per-band normalized, and packed into a fixed-length sequence
    (300 *g* + 350 *r* + 50 *i* = 700).  Shorter sequences are zero-padded.
    Pass a ``band_groups`` dict (e.g. ``{"g": 0, "r": 1, "i": 2}``) to use
    string band labels instead of integers.

    .. important::

        Observation times **must be in Modified Julian Date (MJD)**.  The model
        subtracts a fixed offset of 58 000 during preprocessing, so arbitrary
        time units may produce incorrect embeddings.  The model was pretrained
        on **ZTF DR16** (Zubercal DR16 × Gaia DR3), which covers
        **MJD 58 194 – 59 951** (roughly 2018 Feb – 2023 Jan).

    Parameters
    ----------
    session :
        An ``onnxruntime.InferenceSession`` for the AstraCLR ONNX model.
    band_groups : Mapping, list of Mapping, or None, optional
        Band label → model integer mapping.  When ``None`` (default), the
        caller must supply integer band indices (0=*g*, 1=*r*, 2=*i*).
        Please note that some ZTF data products, e.g. "ordinal" ZTF data
        releases (non-zubercal) would have `filterid` field encoded as
        (1=*g*, 2=*r*, 3=*i*), please use ``{1: 0, 2: 1, 3: 2}`` for such
        cases, and ``{"g": 0, "r": 1, "i": 2}`` for ZTF string band labels.
    allow_extra_bands : bool, optional
        If ``False`` (default), raises when the input contains bands not in
        ``band_groups`` (or not in 0–2 when ``band_groups`` is ``None``).
    reduction : str, list of str, or Reduction, optional
        Strategy for selecting up to the fixed number of observations per band
        before zero-padding.  Defaults to ``"beginning"`` (first observations).
        Any :class:`~light_curve.embed.SingleSubsampleReduction` or
        :class:`~light_curve.embed.MultipleReductions` thereof works well;
        :class:`~light_curve.embed.NonOverlappingWindows` requires each band to
        produce the same number of windows.
    reduction_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`reduction_from_str`.

    Examples
    --------
    >>> import numpy as np
    >>> from light_curve.embed import AstraCLR
    >>> model = AstraCLR.from_hf(
    ...     band_groups={"g": 0, "r": 1, "i": 2},
    ... )
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> mjd = np.sort(rng.uniform(58_194, 59_951, n)).astype(np.float64)
    >>> mag = rng.normal(17, 0.5, n).astype(np.float32)
    >>> magerr = np.full(n, 0.02, dtype=np.float32)
    >>> band = np.array(["g", "r", "i"])[rng.integers(0, 3, n)]
    >>> embedding = model(mjd, mag, magerr, band)
    >>> embedding.shape
    (1, 1, 1, 512)

    Model license
    -------------
    MIT.

    References
    ----------
    Majumder et al. (2026, in prep)

    https://huggingface.co/light-curve/astra-clr
    """

    n_model_bands: int = 3
    seq_sizes: list[int] = [300, 350, 50]
    embed_dim: int = 512
    hf_repo: str = "light-curve/astra-clr"
    hf_filename: str = "astra_clr.onnx"
    model_outputs: frozenset[str] = frozenset({"mean"})

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        band_groups: Mapping | Sequence[Mapping] | None = None,
        allow_extra_bands: bool = False,
        reduction: str | list[str] | Reduction = "beginning",
        reduction_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            session,
            band_groups=band_groups,
            allow_extra_bands=allow_extra_bands,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
        )

    @classmethod
    def from_hf(
        cls,
        *,
        band_groups: Mapping | Sequence[Mapping] | None = None,
        allow_extra_bands: bool = False,
        reduction: str | list[str] | Reduction = "beginning",
        reduction_kwargs: dict[str, object] | None = None,
        ort_session_kwargs: dict[str, object] | None = None,
    ) -> Self:
        """Load the model from the HuggingFace Hub.

        Downloads (and caches) the ONNX model from
        ``https://huggingface.co/light-curve/astra-clr``, creates an
        ``onnxruntime.InferenceSession``, and returns a ready-to-use instance.

        Parameters
        ----------
        band_groups : Mapping, list of Mapping, or None, optional
            Band label → model integer mapping.  When ``None`` (default), the
            caller must supply integer band indices (0=*g*, 1=*r*, 2=*i*).
            Please note that some ZTF data products, e.g. "ordinal" ZTF data
            releases (non-zubercal) would have `filterid` field encoded as
            (1=*g*, 2=*r*, 3=*i*), please use ``{1: 0, 2: 1, 3: 2}`` for such
            cases, and ``{"g": 0, "r": 1, "i": 2}`` for ZTF string band labels.
        allow_extra_bands : bool, optional
            Silently ignore observations with unknown band labels when
            ``True``.  Default is ``False``.
        reduction : str, list of str, or Reduction, optional
            Windowing / subsampling strategy per band.  Default is
            ``"beginning"`` (first observations).
        reduction_kwargs : dict or None, optional
            Extra keyword arguments forwarded to :func:`reduction_from_str`
            when ``reduction`` is given as a string.
        ort_session_kwargs : dict or None, optional
            Keyword arguments forwarded to ``onnxruntime.InferenceSession``:
            ``"sess_options"``, ``"providers"``, ``"provider_options"``.

        Returns
        -------
        AstraCLR
            Instance with a live ONNX inference session.

        Raises
        ------
        ImportError
            If ``huggingface_hub`` is not installed.
        ImportError
            If no ``onnxruntime`` variant is installed.
        """
        return super().from_hf(
            cls.hf_filename,
            band_groups=band_groups,
            allow_extra_bands=allow_extra_bands,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
            ort_session_kwargs=ort_session_kwargs,
        )

    def __call__(
        self,
        mjd: ArrayLike,
        mag: ArrayLike,
        magerr: ArrayLike,
        band: ArrayLike,
    ) -> np.ndarray:
        """Embed a ZTF multi-band light curve.

        Parameters
        ----------
        mjd : array-like, shape ``(n,)``
            Observation times in **Modified Julian Date (MJD)**.  The model
            subtracts a fixed offset of 58 000, so arbitrary time units will
            produce incorrect embeddings.  The model was pretrained on ZTF DR16,
            covering MJD 58 194 – 59 951.
        mag : array-like, shape ``(n,)``
            Magnitudes.
        magerr : array-like, shape ``(n,)``
            Magnitude uncertainties (must be positive).
        band : array-like, shape ``(n,)``
            Band labels.  When ``band_groups`` is ``None``, must be integers
            0 (*g*), 1 (*r*), or 2 (*i*).  Otherwise, must match the keys
            defined in ``band_groups``.

        Returns
        -------
        np.ndarray, shape ``(n_band_groups, n_subsamples, 1, 512)``
            Embedding tensor.  ``n_band_groups`` is 1 when ``band_groups`` is
            ``None`` or a single dict.  ``n_subsamples`` equals the number of
            windows produced by the reduction (1 for single-window reductions;
            *n* for :class:`~light_curve.embed.MultipleReductions` with *n*
            strategies).  The ``seq_size`` axis is always 1 (a single pooled
            vector per light curve).
        """
        return super().__call__(mjd, mag, magerr, band=band)

    def preprocess_single_band(
        self,
        mjd: ArrayLike,
        mag: ArrayLike,
        magerr: ArrayLike,
        *,
        band_idx: int,
        seq_size: int,
    ) -> AstraCLRInputs:
        """Preprocess one band's observations into AstraCLR input tensors.

        Parameters
        ----------
        mjd : array-like, shape ``(n,)``
            Observation times in **Modified Julian Date (MJD)**.  58 000 is
            subtracted before feeding into the model.
        mag : array-like, shape ``(n,)``
            Magnitudes.
        magerr : array-like, shape ``(n,)``
            Magnitude uncertainties.
        band_idx : int
            Model band index (0=*g*, 1=*r*, 2=*i*).
        seq_size : int
            Maximum sequence length for this band.

        Returns
        -------
        AstraCLRInputs
            Tensors with shapes ``(n_subsamples, seq_size, 1)`` for 3-D fields
            and ``(n_subsamples, seq_size)`` for masks.
        """
        mjd = np.asarray(mjd, dtype=np.float64)
        mag = np.asarray(mag, dtype=np.float32)
        magerr = np.asarray(magerr, dtype=np.float32)

        mjd_win, mag_win, magerr_win, bool_mask_win = self.reduction.preprocess_lc(mjd, mag, magerr, seq_size=seq_size)

        n_subsamples = mjd_win.shape[0]

        norm_mag = np.zeros_like(mag_win)
        for s in range(n_subsamples):
            valid = bool_mask_win[s]
            if valid.any():
                m = mag_win[s, valid]
                e = magerr_win[s, valid]
                norm_mag[s, valid] = m - np.average(m, weights=e**-2)
        norm_mag = np.asarray(norm_mag, dtype=np.float32)

        norm_mjd = np.where(bool_mask_win, (mjd_win - _MJD_OFFSET).astype(np.float32), 0.0)
        band_info = np.where(bool_mask_win, _LG_EFF_WAVE[band_idx], 0.0).astype(np.float32)
        float_mask = (~bool_mask_win).astype(np.float32)

        return AstraCLRInputs(
            bool_mask=bool_mask_win,
            input=norm_mag[:, :, np.newaxis],
            times=norm_mjd[:, :, np.newaxis],
            band_info=band_info[:, :, np.newaxis],
            mask=float_mask,
        )

    def predict_tensors(self, tensors: AstraCLRInputs) -> np.ndarray:
        """Run the ONNX model and return reduced embeddings.

        Parameters
        ----------
        tensors : AstraCLRInputs
            As returned by :meth:`preprocess_lc`.

        Returns
        -------
        np.ndarray, shape ``(n_subsamples, 1, 512)``
            One 512-d embedding per subsample, with a unit ``seq_size`` axis
            for compatibility with the 4-D output convention.  For
            :class:`~light_curve.embed.NonOverlappingWindows` the windows are
            averaged to a single embedding.
        """
        (raw,) = self.session.run(None, tensors.asdict())
        # raw shape: (n_subsamples, 512)
        embedding = raw[:, np.newaxis, :]  # (n_subsamples, 1, 512)
        return self.reduction.reduce_embeddings(embedding, tensors, output="mean")
