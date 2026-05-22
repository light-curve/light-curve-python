from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from light_curve.embed.input_tensors import InputTensors
from light_curve.embed.model import ImplicitMultiBandModel

if TYPE_CHECKING:
    from typing import Self


_MJD_OFFSET = 58_000.0
_LG_EFF_WAVE = {
    "g": np.log10(4746.48),
    "r": np.log10(6366.38),
    "i": np.log10(7829.03),
}


@dataclass
class AstraCLRInputs(InputTensors):
    """Input tensors for AstraCLR.

    All arrays have batch size 1.  Mask convention: 1 = padded, 0 = real
    (opposite of Astromer).
    """

    input: np.ndarray = field(kw_only=True)
    times: np.ndarray = field(kw_only=True)
    band_info: np.ndarray = field(kw_only=True)
    mask: np.ndarray = field(kw_only=True)


class AstraCLR(ImplicitMultiBandModel):
    """AstraCLR multiband transformer embedding model.

    AstraCLR is a contrastive-learning encoder for ZTF (g, r, i) multi-band
    photometric light curves.  It maps a light curve to a 512-dimensional
    embedding via a transformer architecture trained with a contrastive
    objective.

    The model concatenates per-band observation windows into a fixed-length
    sequence of 700 elements (300 g + 350 r + 50 i), sorted chronologically
    within each band.  Magnitudes are inverse-variance-weighted mean-subtracted
    per band.

    Use :meth:`from_hf` to download and load the model directly.

    Parameters
    ----------
    session :
        An ``onnxruntime.InferenceSession`` for the AstraCLR ONNX model.

    Examples
    --------
    >>> import numpy as np
    >>> import light_curve.embed as lce
    >>> model = lce.AstraCLR.from_hf()
    >>> n = 200
    >>> rng = np.random.default_rng(0)
    >>> time = np.sort(rng.uniform(58000, 60500, n))
    >>> mag = rng.uniform(17, 21, n).astype(np.float32)
    >>> magerr = rng.uniform(0.01, 0.15, n).astype(np.float32)
    >>> band = rng.choice(["g", "r", "i"], n)
    >>> embedding = model(time, mag, magerr, band)
    >>> embedding.shape
    (1, 1, 1, 512)

    Model license
    -------------
    MIT (© 2025 Torsha Majumder).

    References
    ----------
    Majumder et al., 2026, in prep.
    https://github.com/TorshaMajumder/astra
    """

    seq_per_band: list[tuple[str, int]] = [("g", 300), ("r", 350), ("i", 50)]
    hf_repo: str = "light-curve/astra-clr"
    hf_filename: str = "astra_clr.onnx"
    model_outputs: frozenset[str] = frozenset({"mean"})

    @classmethod
    def from_hf(
        cls,
        *,
        ort_session_kwargs: dict[str, object] | None = None,
    ) -> Self:
        """Load AstraCLR from the HuggingFace Hub.

        Downloads (and caches) ``astra_clr.onnx`` from
        ``light-curve/astra-clr``, creates an ONNX inference session, and
        returns a ready-to-use instance.

        Parameters
        ----------
        ort_session_kwargs : dict or None, optional
            Additional keyword arguments forwarded to
            ``onnxruntime.InferenceSession``: ``"sess_options"``,
            ``"providers"``, ``"provider_options"``.

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
        return super().from_hf(cls.hf_filename, ort_session_kwargs=ort_session_kwargs)

    def preprocess_lc(
        self,
        time: ArrayLike,
        mag: ArrayLike,
        magerr: ArrayLike,
        band: ArrayLike,
    ) -> AstraCLRInputs:
        """Preprocess a light curve into AstraCLR model input tensors.

        Sorts observations chronologically within each band, selects the first
        ``SEQ_PER_BAND[band]`` observations, subtracts the
        inverse-variance-weighted mean magnitude per band, and assembles the
        fixed-layout sequence.

        Parameters
        ----------
        time : array-like, shape ``(n,)``
            Observation times in MJD.
        mag : array-like, shape ``(n,)``
            Magnitudes.
        magerr : array-like, shape ``(n,)``
            Magnitude uncertainties.
        band : array-like, shape ``(n,)``
            Band labels: ``"g"``, ``"r"``, or ``"i"``.

        Returns
        -------
        AstraCLRInputs
        """
        time = np.asarray(time, dtype=np.float64)
        mag = np.asarray(mag, dtype=np.float32)
        magerr = np.asarray(magerr, dtype=np.float32)
        band = np.asarray(band)

        norm_mag = np.zeros(self.seq_len, dtype=np.float32)
        norm_time = np.zeros(self.seq_len, dtype=np.float32)
        band_info_arr = np.zeros(self.seq_len, dtype=np.float32)
        mask = np.ones(self.seq_len, dtype=np.float32)  # 1 = padded, 0 = real
        bool_mask = np.zeros(self.seq_len, dtype=bool)

        for b, _n, offset, t_slot, m_slot, e_slot in self._iterate_band_slots(time, mag, magerr, band):
            n_real = len(t_slot)
            if n_real > 0:
                weights = e_slot**-2
                weighted_mean = np.average(m_slot, weights=weights)
                norm_mag[offset : offset + n_real] = m_slot - weighted_mean
                norm_time[offset : offset + n_real] = t_slot - _MJD_OFFSET
                band_info_arr[offset : offset + n_real] = _LG_EFF_WAVE[b]
                mask[offset : offset + n_real] = 0.0
                bool_mask[offset : offset + n_real] = True

        return AstraCLRInputs(
            input=norm_mag.reshape(1, self.seq_len, 1),
            times=norm_time.reshape(1, self.seq_len, 1),
            band_info=band_info_arr.reshape(1, self.seq_len, 1),
            mask=mask.reshape(1, self.seq_len),
            bool_mask=bool_mask.reshape(1, self.seq_len),
        )

    def predict_tensors(self, tensors: AstraCLRInputs) -> np.ndarray:
        """Run the ONNX model on pre-processed tensors.

        Parameters
        ----------
        tensors : AstraCLRInputs
            As returned by :meth:`preprocess_lc`.

        Returns
        -------
        np.ndarray, shape ``(1, 1, 1, 512)``
            Embedding in ``(n_bands, n_subsamples, seq_size, embed_dim)`` layout.
        """
        (embedding,) = self.session.run(["mean"], tensors.asdict())
        return embedding.reshape(1, 1, -1)
