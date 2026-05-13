from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from light_curve.embed.input_tensors import InputTensors

if TYPE_CHECKING:
    from typing import Self


class Reduction(ABC):
    """Abstract base for strategies that map a variable-length light curve to fixed-length sequences.

    Subclasses implement :meth:`subsample_lc` (windowing / subsampling logic) and
    :meth:`reduce_embeddings` (how per-window embeddings are aggregated back to one
    embedding per light curve).
    """

    def _mask(self, *, array_size: int, seq_size: int) -> np.ndarray:
        padding_size = max(seq_size - array_size, 0)
        values = np.ones(array_size, dtype=bool)
        padding = np.zeros(padding_size, dtype=bool)
        return np.r_[values, padding]

    def _pad(self, array: np.ndarray, seq_size: int) -> np.ndarray:
        if array.size >= seq_size:
            return array[:seq_size]
        return np.r_[array, np.zeros(seq_size - array.size, dtype=array.dtype)]

    def preprocess_lc(self, *arrays: ArrayLike, seq_size: int) -> tuple[np.ndarray, ...]:
        """Subsample, pad, and mask a light curve.

        Parameters
        ----------
        *arrays : array-like
            1-D arrays of equal length (e.g. time, magnitude).
        seq_size : int
            Target sequence length; shorter windows are zero-padded.

        Returns
        -------
        tuple of np.ndarray
            One stacked array per input plus a boolean mask, each of shape
            ``(n_subsamples, seq_size)``.  The mask is ``True`` for real
            observations and ``False`` for padding.

        Raises
        ------
        ValueError
            If any input array is not 1-D.
        """
        arrays = np.broadcast_arrays(*arrays)
        if arrays[0].ndim != 1:
            raise ValueError(f"Inputs must be single dimensional, {arrays[0].ndim} is provided")

        subsamples = self.subsample_lc(*arrays, seq_size=seq_size)
        sequences = []
        for subsample in subsamples:
            mask = self._mask(array_size=subsample[0].size, seq_size=seq_size)
            sequence = tuple(self._pad(array, seq_size) for array in subsample) + (mask,)
            sequences.append(sequence)

        sequences = tuple(np.stack(arrays) for arrays in zip(*sequences))

        return sequences

    @abstractmethod
    def subsample_lc(self, *arrays: np.ndarray, seq_size: int) -> list[tuple[np.ndarray, ...]]:
        """Split or subsample the light curve into one or more windows.

        Parameters
        ----------
        *arrays : np.ndarray
            1-D arrays of equal length.
        seq_size : int
            Maximum number of observations per window.

        Returns
        -------
        list of tuple of np.ndarray
            Each element is a tuple of arrays (one per input), containing at most
            ``seq_size`` observations.
        """
        raise NotImplementedError

    @abstractmethod
    def reduce_embeddings(self, embeddings: ArrayLike, tensors: InputTensors, *, output: str) -> ArrayLike:
        """Aggregate per-window embeddings into a single embedding per light curve.

        Parameters
        ----------
        embeddings : array-like, shape ``(n_windows, seq_size, embed_dim)``
            Raw per-window embeddings from the model.  ``seq_size`` is 1 for
            aggregated (mean / max) outputs and the full sequence length for the
            ``"sequence"`` output.
        tensors : InputTensors
            Preprocessed input tensors as returned by
            :meth:`EmbeddingSession.preprocess_lc`.
        output : str
            The model output name (e.g. ``"mean"``, ``"max"``, ``"sequence"``).
            Implementations use this to select the appropriate aggregation strategy
            rather than inferring it from array shapes.

        Returns
        -------
        array-like
            Reduced embeddings, shape depends on the concrete implementation.
        """
        raise NotImplementedError


class SingleSubsampleReduction(Reduction, ABC):
    """Base for strategies that produce exactly one window per light curve."""

    def subsample_lc(self, *arrays: np.ndarray, seq_size: int) -> list[tuple[np.ndarray, ...]]:
        """Return a single-element list wrapping :meth:`single_subsample_lc`.

        Parameters
        ----------
        *arrays : np.ndarray
            1-D arrays of equal length.
        seq_size : int
            Maximum observations per window.

        Returns
        -------
        list of tuple of np.ndarray
            A one-element list containing the subsampled arrays.
        """
        return [self.single_subsample_lc(*arrays, seq_size=seq_size)]

    @abstractmethod
    def single_subsample_lc(self, *arrays: np.ndarray, seq_size: int) -> tuple[np.ndarray, ...]:
        """Return one subsampled window of at most ``seq_size`` observations.

        Parameters
        ----------
        *arrays : np.ndarray
            1-D arrays of equal length.
        seq_size : int
            Maximum number of observations to keep.

        Returns
        -------
        tuple of np.ndarray
            Subsampled arrays, each of length ``<= seq_size``.
        """
        raise NotImplementedError

    def reduce_embeddings(self, embeddings: ArrayLike, tensors: InputTensors, *, output: str) -> ArrayLike:
        """Return embeddings unchanged (single window — no aggregation needed).

        Parameters
        ----------
        embeddings : array-like
            Per-window embeddings from the model.
        tensors : InputTensors
            Unused; accepted for interface compatibility.
        output : str
            Unused; accepted for interface compatibility.

        Returns
        -------
        array-like
            The input unchanged.
        """
        return embeddings


class Beginning(SingleSubsampleReduction):
    """Select the chronologically first ``seq_size`` observations of the light curve."""

    def single_subsample_lc(self, *arrays: np.ndarray, seq_size: int) -> tuple[np.ndarray, ...]:
        """Return the leading ``seq_size`` elements of each array.

        Parameters
        ----------
        *arrays : np.ndarray
            1-D arrays of equal length.
        seq_size : int
            Number of observations to keep from the start.

        Returns
        -------
        tuple of np.ndarray
            First ``seq_size`` elements of each input array.
        """
        return tuple(array[:seq_size] for array in arrays)


class End(SingleSubsampleReduction):
    """Select the chronologically last ``seq_size`` observations of the light curve."""

    def single_subsample_lc(self, *arrays: np.ndarray, seq_size: int) -> tuple[np.ndarray, ...]:
        """Return the trailing ``seq_size`` elements of each array.

        Parameters
        ----------
        *arrays : np.ndarray
            1-D arrays of equal length.
        seq_size : int
            Number of observations to keep from the end.

        Returns
        -------
        tuple of np.ndarray
            Last ``seq_size`` elements of each input array.
        """
        return tuple(array[-seq_size:] for array in arrays)


class RandomSubsample(SingleSubsampleReduction):
    """Draw ``seq_size`` observations uniformly at random without replacement.

    Parameters
    ----------
    rng : int, np.random.Generator, or None
        Seed or generator for reproducible sampling.
    """

    def __init__(self, *, rng: int | np.random.Generator | None) -> None:
        super().__init__()
        self.rng = np.random.default_rng(rng)

    def single_subsample_lc(self, *arrays: np.ndarray, seq_size: int) -> tuple[np.ndarray, ...]:
        """Return a random subsample of at most ``seq_size`` observations, in original order.

        Parameters
        ----------
        *arrays : np.ndarray
            1-D arrays of equal length.
        seq_size : int
            Maximum number of observations to sample.

        Returns
        -------
        tuple of np.ndarray
            ``min(len, seq_size)`` randomly selected observations, sorted by
            original index so temporal order is preserved.
        """
        array_size = arrays[0].size
        if array_size <= seq_size:
            return arrays

        indices = self.rng.choice(array_size, size=seq_size, replace=False)
        indices.sort()
        return tuple(array[indices] for array in arrays)


class NonOverlappingWindows(Reduction):
    """Split the light curve into consecutive non-overlapping windows of ``seq_size`` observations.

    A light curve of length *L* yields ``ceil(L / seq_size)`` windows; the last window
    may be shorter than ``seq_size`` and is zero-padded.  Per-window embeddings are
    averaged to produce a single embedding per light curve.
    """

    def subsample_lc(self, *arrays: np.ndarray, seq_size: int) -> list[tuple[np.ndarray, ...]]:
        """Yield consecutive slices of length ``seq_size``.

        Parameters
        ----------
        *arrays : np.ndarray
            1-D arrays of equal length.
        seq_size : int
            Window size.

        Returns
        -------
        list of tuple of np.ndarray
            ``ceil(len / seq_size)`` windows, each a tuple of sliced arrays.
        """
        array_size = arrays[0].size

        subsamples = []
        for start in range(0, array_size, seq_size):
            end = start + seq_size
            subsample = tuple(array[start:end] for array in arrays)
            subsamples.append(subsample)

        return subsamples

    def reduce_embeddings(self, embeddings: np.ndarray, tensors: InputTensors, *, output: str) -> np.ndarray:
        """Reduce per-window embeddings to a single representation.

        For aggregated outputs (``output != "sequence"``) the window embeddings
        are averaged, yielding shape ``(1, 1, embed_dim)``.

        For ``output == "sequence"`` a masked mean is computed across windows
        for each timestep position, yielding shape ``(1, seq_size, embed_dim)``
        regardless of how many windows the light curve was split into.

        Parameters
        ----------
        embeddings : np.ndarray, shape ``(n_windows, seq_size, embed_dim)``
            Per-window embeddings.
        tensors : InputTensors
            Preprocessed input tensors; ``tensors.bool_mask`` (shape
            ``(n_windows, seq_size)``) identifies valid vs. padded positions
            for the ``"sequence"`` output.
        output : str
            Model output name.  Determines aggregation strategy.

        Returns
        -------
        np.ndarray, shape ``(1, 1, embed_dim)`` or ``(1, seq_size, embed_dim)``
            For mean / max: mean over windows, shape ``(1, 1, embed_dim)``.
            For sequence: masked mean over windows per timestep, shape
            ``(1, seq_size, embed_dim)``.
        """
        if output != "sequence":
            return np.mean(embeddings, axis=0)[np.newaxis, ...]
        # sequence output: masked mean over windows per timestep → (1, seq_size, embed_dim)
        n_valid = np.maximum(tensors.bool_mask.sum(axis=0), 1)  # (seq_size,)
        window_sum = (embeddings * tensors.bool_mask[:, :, np.newaxis]).sum(axis=0)  # (seq_size, embed_dim)
        return (window_sum / n_valid[:, np.newaxis])[np.newaxis, ...]


class MultipleReductions(Reduction):
    """Apply several :class:`SingleSubsampleReduction` strategies in parallel.

    Each strategy produces one window; embeddings are stacked along the subsample
    axis rather than aggregated, giving one embedding per strategy.

    Parameters
    ----------
    reductions : list of SingleSubsampleReduction
        Ordered list of strategies to apply.

    Raises
    ------
    ValueError
        If any element of ``reductions`` is not a
        :class:`SingleSubsampleReduction`.
    """

    def __init__(
        self,
        reductions: list[SingleSubsampleReduction],
    ) -> None:
        super().__init__()
        for r in reductions:
            if not isinstance(r, SingleSubsampleReduction):
                raise ValueError(
                    f"Reduction '{r}' is not a subsampling reduction; "
                    "currently only subsampling reductions can be used in multiple reductions"
                )
        self.reductions = reductions

    @classmethod
    def from_strings(
        cls,
        reductions: list[str],
        **kwargs,
    ) -> Self:
        """Construct from a list of strategy name strings.

        Parameters
        ----------
        reductions : list of str
            Strategy names recognised by :func:`reduction_from_str`.
        **kwargs
            Forwarded to each strategy constructor.  If ``rng`` is an integer
            seed it is converted to a :class:`numpy.random.Generator` first so
            that each stochastic strategy gets an independent random stream.

        Returns
        -------
        MultipleReductions
            Instance wrapping the instantiated strategies.
        """
        # In the cases where rng is an integer (seed), we should convert to a Generator first, so we don't reuse the
        # same seed across multiple reductions. Reusing the same rng is ok: each call mutates it.
        if "rng" in kwargs:
            kwargs["rng"] = np.random.default_rng(kwargs["rng"])
        return cls(
            reductions=[reduction_from_str(s, **kwargs) for s in reductions],
        )

    def subsample_lc(self, *arrays: np.ndarray, seq_size: int) -> list[tuple[np.ndarray, ...]]:
        """Apply each strategy and return one window per strategy.

        Parameters
        ----------
        *arrays : np.ndarray
            1-D arrays of equal length.
        seq_size : int
            Maximum observations per window.

        Returns
        -------
        list of tuple of np.ndarray
            One element per strategy, each a tuple of subsampled arrays.
        """
        subsamples = []
        for reduction in self.reductions:
            subsamples.append(reduction.single_subsample_lc(*arrays, seq_size=seq_size))
        return subsamples

    def reduce_embeddings(self, embeddings: ArrayLike, tensors: InputTensors, *, output: str) -> ArrayLike:
        """Return embeddings unchanged — one per strategy, already stacked.

        Parameters
        ----------
        embeddings : array-like
            Per-strategy embeddings from the model.
        tensors : InputTensors
            Unused; accepted for interface compatibility.
        output : str
            Unused; accepted for interface compatibility.

        Returns
        -------
        array-like
            The input unchanged.
        """
        return embeddings


def reduction_from_str(s: str | list[str], **kwargs) -> Reduction:
    """Instantiate a :class:`Reduction` from a name string or list of name strings.

    Parameters
    ----------
    s : str or list of str
        Strategy name or list of names.  Recognised values (case-insensitive,
        underscores treated as hyphens): ``"beginning"``, ``"end"``,
        ``"random-subsample"``, ``"non-overlapping-windows"``.  A list with more
        than one entry produces a :class:`MultipleReductions`.
    **kwargs
        Forwarded to the strategy constructor (e.g. ``rng`` for
        :class:`RandomSubsample`).

    Returns
    -------
    Reduction
        The instantiated strategy.

    Raises
    ------
    ValueError
        If the name is not recognised, or if ``rng`` is missing for
        ``"random-subsample"``.
    """
    if isinstance(s, list):
        if len(s) != 1:
            return MultipleReductions.from_strings(s, **kwargs)
        s = s[0]

    match s.lower().replace("_", "-"):
        case "beginning":
            return Beginning()
        case "end":
            return End()
        case "random-subsample":
            try:
                rng = kwargs["rng"]
            except KeyError:
                raise ValueError("rng must be provided for random subsample reduction")
            return RandomSubsample(rng=rng)
        case "non-overlapping-windows":
            return NonOverlappingWindows()
        case _:
            raise ValueError(f"Unknown reduction '{s}'")
