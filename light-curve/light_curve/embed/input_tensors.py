from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class InputTensors:
    """Base tensor container produced by :meth:`EmbeddingSession.preprocess_lc`.

    All subclasses carry a boolean ``bool_mask`` identifying valid (non-padded)
    timesteps, shape ``(n_windows, seq_size)``.
    """

    bool_mask: np.ndarray = field(kw_only=True)

    def asdict(self) -> dict[str, np.ndarray]:
        d = asdict(self)
        d.pop("bool_mask")
        return d


def concat_input_tensors(tensors: list[InputTensors]) -> InputTensors:
    ty = type(tensors[0])
    arrays = defaultdict(list)
    for t in tensors:
        assert isinstance(t, ty), "All tensors must be of the same type"
        for k, v in asdict(t).items():
            arrays[k].append(v)
    arrays = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
    return ty(**arrays)
