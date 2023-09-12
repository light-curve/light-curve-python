"""Re-implementation of iminuit.cost.LeastSquares with an arbitrary data format"""

from typing import Callable, Dict, Tuple

import numpy as np

try:
    from iminuit import Minuit
except ImportError:
    LeastSquares = None
else:
    # Following https://iminuit.readthedocs.io/en/stable/notebooks/generic_least_squares.html
    class LeastSquares:
        errordef = Minuit.LEAST_SQUARES

        def __init__(self, model: Callable, parameters: Dict[str, Tuple[float, float]], *, x, y, yerror):
            self.model = model
            self.x = x
            self.y = y
            self.yerror = yerror
            self._parameters = parameters

        @property
        def ndata(self):
            return len(self.y)

        def __call__(self, *par):
            ym = self.model(self.x, *par)
            return np.sum(np.square((self.y - ym) / self.yerror))


__all__ = ["LeastSquares"]
