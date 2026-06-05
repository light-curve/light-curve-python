from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from collections import Counter
from enum import IntEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from light_curve.embed.input_tensors import InputTensors, concat_input_tensors
from light_curve.embed.reduction import Reduction, reduction_from_str
from light_curve.light_curve_py.warnings import warn_experimental

if TYPE_CHECKING:
    from typing import Self

    import onnxruntime as ort


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
    reduction_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`reduction_from_str`
        when ``reduction`` is given as a string.
    """

    hf_repo: str | None
    model_outputs: frozenset[str]

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        reduction: str | list[str] | Reduction,
        reduction_kwargs: dict[str, object] | None = None,
    ) -> None:
        warn_experimental(
            f"{self.__class__.__module__}.{self.__class__.__name__} is experimental and may change in future versions"
        )
        self.session = session

        if reduction_kwargs is None:
            reduction_kwargs = {}
        if not isinstance(reduction, Reduction):
            reduction = reduction_from_str(reduction, **reduction_kwargs)
        self.reduction = reduction

    @classmethod
    def from_hf(
        cls,
        filename: str,
        *,
        ort_session_kwargs: dict[str, object] | None = None,
        **kwargs,
    ) -> Self:
        """Load a model from the HuggingFace Hub.

        Downloads (and caches) the ONNX model file, creates an
        ``onnxruntime.InferenceSession``, and returns a ready-to-use instance.

        Parameters
        ----------
        filename: str
            Path to the model file inside ``self.hf_repo``.
        ort_session_kwargs: dict[str, object] or None, optional
            Options to pass to the ``onnxruntime.InferenceSession`` constructor:
            "sess_options", "providers", "provider_options".
        **kwargs
            Forwarded verbatim to the class constructor

        Returns
        -------
        instance of the calling class
            Instance with a live ONNX inference session.

        Raises
        ------
        ImportError
            If ``huggingface_hub`` is not installed.
        ImportError
            If no ``onnxruntime`` variant is installed.
        """
        model_path = _hf_hub_download_cached(cls.hf_repo, filename)

        if ort_session_kwargs is None:
            ort_session_kwargs = {}
        session = create_onnx_session(model_path, **ort_session_kwargs)

        return cls(session=session, **kwargs)

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
    reduction_kwargs : dict, optional
        Extra kwargs forwarded to :func:`reduction_from_str`.
    """

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        bands: Sequence[str | int] | None = None,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        reduction_kwargs: dict[str, object] | None = None,
    ) -> None:
        super().__init__(
            session,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
        )
        self.bands = bands

    @abstractmethod
    def __call__(self, *arrays: ArrayLike, band: ArrayLike | None = None) -> np.ndarray:
        """Embed a light curve.

        Parameters
        ----------
        *arrays : array-like, shape ``(n,)``
            Raw light curve arrays (e.g. time, magnitude).
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
            embed = super().__call__(*arrays)
            return np.expand_dims(embed, axis=Dim.BAND)

        embeddings = []
        for band_name in self.bands:
            band_idx = band == band_name
            embed_band = super().__call__(*(arr[band_idx] for arr in arrays))
            embeddings.append(np.expand_dims(embed_band, axis=Dim.BAND))
        return np.concatenate(embeddings, axis=Dim.BAND)


class MultiBandModel(EmbeddingSession, ABC):
    """Embedding model that receives all photometric bands in a single forward pass.

    Unlike :class:`SingleBandModel`, model may accept multiple photometric bands.

    ``band_groups`` controls how user-supplied band labels are mapped to the
    integer indices the model expects:

    * ``None`` (default) — the caller must already supply integer indices in
      ``[0, ..., max(valid_model_bands)]``; all observations are fed to the model
      as one group and the output has ``n_band_groups = 1``.
    * A ``dict`` mapping input labels → model integers — the mapping is applied
      before inference; all observations are still fed as one group
      (``n_band_groups = 1``).
    * A ``list`` of such dicts — each dict defines an independent group; the
      model runs once per group and the results are stacked along the band axis
      (``n_band_groups = len(band_groups)``).  Input labels must be unique across the
      entire list.

    Parameters
    ----------
    session :
        ONNX inference session.
    band_groups : Mapping, list of Mapping, or None, optional
        Band label → model integer mapping(s).  See class docstring for details.
    allow_extra_bands : bool, optional
        If ``False`` (default), :meth:`__call__` raises :exc:`ValueError` when
        the input contains band labels not covered by ``band_groups`` (or not in
        ``valid_model_bands`` when ``band_groups`` is ``None``).  Set to ``True``
        to silently ignore observations with unknown band labels.
    reduction : str, list of str, or Reduction
        Windowing / subsampling strategy.
    reduction_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`reduction_from_str`.
    """

    n_model_bands: int

    def __init__(
        self,
        session: ort.InferenceSession,
        *,
        band_groups: Mapping | Sequence[Mapping] | None = None,
        allow_extra_bands: bool = False,
        reduction: str | list[str] | Reduction = "non-overlapping-windows",
        reduction_kwargs: dict[str, object] | None = None,
    ):
        super().__init__(
            session,
            reduction=reduction,
            reduction_kwargs=reduction_kwargs,
        )
        self.bands_groups = band_groups
        self.allow_extra_bands = allow_extra_bands

        if band_groups is None:
            self.band_mappings = None
            self.defined_input_bands = set(range(self.n_model_bands))
            return

        if isinstance(band_groups, Sequence):
            keys_counter = Counter()
            uniq_values = set()
            band_mappings = []
            for band_group in band_groups:
                keys_counter.update(band_group.keys())
                uniq_values.update(band_group.values())
                band_mappings.append((frozenset(band_group.keys()), np.vectorize(band_group.get)))
            if keys_counter.total() > len(keys_counter):
                duplicate_keys = [key for key, count in keys_counter.most_common() if count > 1]
                raise ValueError(
                    "If ``band_groups`` is a list ``dict[input_band, model_map]``, "
                    "then all ``input_band``s must be unique across the list. "
                    f"Duplicate keys found: {duplicate_keys}"
                )
            defined_input_bands = set(keys_counter.keys())
        else:
            uniq_values = set(band_groups.values())
            band_mappings = [(frozenset(band_groups.keys()), np.vectorize(band_groups.get))]
            defined_input_bands = set(band_groups.keys())
        if not uniq_values.issubset(range(self.n_model_bands)):
            raise ValueError(
                "if ``band_groups`` is a ``dict[input_band, model_map]``, or a list of such dicts, "
                "then all ``model_map`` values must be in the set of valid model band_groups. "
                f"Invalid values found: {sorted(uniq_values - set(range(self.n_model_bands)))}"
            )

        self.band_mappings = band_mappings
        self.defined_input_bands = defined_input_bands

    def _validate_extra_bands(self, band: ArrayLike):
        if self.allow_extra_bands:
            return
        uniq_input_bands = np.unique(band)
        if not self.defined_input_bands.issuperset(uniq_input_bands):
            raise ValueError(
                "all ``band`` values must be in the defined input bands "
                "(keys of ``band_groups``, or ``valid_model_bands`` when ``band_groups`` is ``None``). "
                f"Invalid values found: {sorted(set(uniq_input_bands) - self.defined_input_bands)}"
            )

    @abstractmethod
    def __call__(self, *arrays: ArrayLike, band: ArrayLike) -> np.ndarray:
        """Embed a light curve.

        Parameters
        ----------
        *arrays : array-like
            Raw light curve arrays (e.g. time, magnitude).
        band : array-like, shape ``(n,)``
            Band labels.  When ``band_groups`` is ``None``, must be integers in
            ``valid_model_bands``; otherwise must match the keys defined in
            ``band_groups``.

        Returns
        -------
        np.ndarray, shape ``(n_band_groups, n_subsamples, seq_size, embed_dim)``
            Embedding tensor.  ``n_band_groups`` is 1 when ``band_groups`` is ``None`` or
            a single dict, and equals the number of dicts when ``band_groups`` is a list.

        Raises
        ------
        ValueError
                If ``band`` contains values not in the defined input band_groups (when ``allow_extra_bands=False``).
        """
        self._validate_extra_bands(band)

        if self.bands_groups is None:
            embed = super().__call__(*(arrays + (band,)))
            return np.expand_dims(embed, axis=Dim.BAND)

        if self.band_mappings is None:
            raise RuntimeError("Logical error: band_mappings is None but band_groups is not None")

        embeddings = []
        for group_keys, band_mapping in self.band_mappings:
            group_mask = np.isin(band, list(group_keys))
            filtered_arrays = tuple(arr[group_mask] for arr in arrays)
            model_band = band_mapping(band[group_mask])
            embed_band_group = super().__call__(*(filtered_arrays + (model_band,)))
            embeddings.append(np.expand_dims(embed_band_group, axis=Dim.BAND))

        return np.concatenate(embeddings, axis=Dim.BAND)


class ImplicitMultiBandModel(MultiBandModel, ABC):
    seq_sizes: list[int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.seq_sizes) != self.n_model_bands:
            raise ValueError(
                f"Length of seq_sizes must match n_model_bands. "
                f"Got len(seq_sizes)={len(self.seq_sizes)} and n_model_bands={self.n_model_bands}."
            )

    def preprocess_lc(self, *arrays_with_band: ArrayLike) -> InputTensors:
        *arrays, band = arrays_with_band
        band = np.asarray(band)

        inputs = []
        n_subsamples = None
        for band_idx, seq_size in enumerate(self.seq_sizes):
            band_mask = band == band_idx
            band_arrays = tuple(np.asarray(arr)[band_mask] for arr in arrays)
            band_tensors = self.preprocess_single_band(*band_arrays, band_idx=band_idx, seq_size=seq_size)
            if n_subsamples is None:
                n_subsamples = band_tensors.bool_mask.shape[0]
            elif band_tensors.bool_mask.shape[0] != n_subsamples:
                raise ValueError(
                    f"Band {band_idx} produced {band_tensors.bool_mask.shape[0]} subsamples "
                    f"but previous bands produced {n_subsamples}."
                )
            inputs.append(band_tensors)

        return concat_input_tensors(inputs, axis=1)

    @abstractmethod
    def preprocess_single_band(self, *arrays: ArrayLike, band_idx: int, seq_size: int) -> InputTensors:
        raise NotImplementedError


_logger = logging.getLogger(__name__)

@lru_cache
def _hf_hub_download_cached(
    repo_id: str,
    filename: str,
    max_attempts: int = 5,
    retry_fallback_seconds: int = 60,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        hf_url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        raise ImportError(
            "huggingface_hub is required to download models from HuggingFace.\n"
            "Install it with:\n"
            "  pip install huggingface-hub\n"
            "Or download the model file directly:\n"
            f"  {hf_url}\n"
            "then load it with:\n"
            "  import onnxruntime as ort\n"
            f'  session=ort.InferenceSession("/path/to/{filename}")'
        ) from exc

    for attempt in range(1, max_attempts + 1):
        try:
            return hf_hub_download(repo_id=repo_id, filename=filename)
        except HfHubHTTPError as exc:
            if exc.response.status_code != 429 or attempt == max_attempts:
                raise
            wait = math.ceil(float(exc.response.headers.get("Retry-After", retry_fallback_seconds - 5))) + 5
        _logger.warning(
            f"HuggingFace rate limit hit for {repo_id}/{filename} "
            f"(attempt {attempt}/{max_attempts}), retrying in {wait} s"
        )
        time.sleep(wait)

    raise RuntimeError("unreachable")


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
