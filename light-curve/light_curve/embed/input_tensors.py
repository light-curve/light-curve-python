from __future__ import annotations

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
