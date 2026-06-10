"""Maximum-likelihood based cost function"""

from typing import Callable, Dict, Tuple

import numpy as np

try:
    from iminuit import Minuit
    from scipy.special import erf
except ImportError:
    MaximumLikelihood = None
else:

    class MaximumLikelihood:
        errordef = Minuit.LIKELIHOOD

        # Coefficient of the soft 1/x barrier that keeps parameters away from their bounds.
        # Kept tiny so the penalty is << 0.5 (the 1-sigma scale of the 0.5*chi^2 NLL) over
        # most of the range and only bites very close to a bound.
        _BARRIER_STRENGTH = 1e-4

        def __init__(
            self,
            model: Callable,
            parameters: Dict[str, Tuple[float, float]],
            upper_mask=None,
            *,
            x,
            y,
            yerror,
            jac: Callable = None,
            prior=None,
        ):
            self.model = model
            self.jac = jac
            self.x = x
            self.y = y
            self.yerror = yerror
            self.upper_mask = upper_mask
            # Optional Gaussian priors: (idx, mean, inv_sigma2) arrays adding
            # 0.5·Σ inv_sigma2·(par[idx] - mean)² to the negative-log-likelihood.
            # None means no priors (the common case, zero overhead).
            self._prior = prior

            self._parameters = parameters
            # FIXME: here we assume the order of dict keys is the same as in parameters array
            self.limits = np.array([_ for _ in parameters.values()])
            self.limits0 = self.limits[:, 0]
            self.limits1 = self.limits[:, 1]
            self.limits_scale = self.limits[:, 1] - self.limits[:, 0]

            self._inv_yerror2 = 1.0 / np.asarray(yerror) ** 2

            # One-slot model cache; see `_model` for the full rationale and scope.
            # Held as a single (par, ym) tuple so it is published with one atomic
            # assignment.
            self._cache = None

        @property
        def ndata(self):
            return len(self.y)

        def _model(self, par):
            # Memoise the most recent model evaluation, keyed on the parameter tuple.
            #
            # WHY THIS EXISTS
            #   Migrad evaluates the cost value (`__call__`) and its gradient (`grad`)
            #   at the *same* parameter point on every step. Both need `model(x, *par)`.
            #   Without this cache the model is computed twice per point; with it, the
            #   second call (whichever of value/gradient comes second) reuses the first
            #   result. This is the entire speed benefit — nothing else depends on it.
            #
            # SCOPE — read before relying on or removing it
            #   * The cache is per-instance state on this cost object, and a *fresh*
            #     MaximumLikelihood (hence a fresh, empty cache) is built for every fit
            #     in BaseRainbowFit._eval_and_get_errors. It is therefore never shared
            #     between light curves / fits — batch and process-parallel fitting are
            #     unaffected because each fit owns its own cache.
            #   * It is a single slot, not a dict: it only ever helps the immediate
            #     value/gradient pair at one point. It is not a general memo table and
            #     will thrash (all misses) if evaluated at many points in a row.
            #
            # CORRECTNESS INVARIANTS — do not break these
            #   * `model(self.x, *par)` must be a pure, deterministic function of `par`.
            #     `self.x` and `self.model` are set at construction and must not be
            #     mutated afterwards, or the cache will return stale results.
            #   * The cached array is returned BY REFERENCE. Callers (`__call__`, `grad`)
            #     only read it. Never mutate a model result in place, or you corrupt the
            #     paired call's view of it.
            #
            # CONCURRENCY
            #   Designed for Migrad's strictly serial evaluation. We still publish the
            #   (par, ym) pair as a single tuple via one assignment, and read it into a
            #   local first, so that IF this instance were ever shared across threads a
            #   reader can only ever see None, the old pair, or this complete new pair —
            #   never a torn mix of one par with another par's ym. (This keeps it safe,
            #   not fast, under sharing — a parallel multi-point optimiser would still
            #   just thrash the single slot.)
            #
            # WHY NOT functools.lru_cache
            #   The working set is exactly one point (the value/gradient pair), so an
            #   lru_cache(maxsize=1) would give the identical hit rate with no benefit, while
            #   `@lru_cache` on this method would key on and retain `self`, leaking every
            #   per-fit cost instance (a fresh one is built per light curve). A per-instance
            #   lru_cache avoids the leak but adds a lock + cache object per fit; this manual
            #   slot is per-instance, GC-clean and lock-free on a hot path.
            cache = self._cache
            if cache is not None and cache[0] == par:
                return cache[1]
            ym = self.model(self.x, *par)
            self._cache = (par, ym)
            return ym

        def __call__(self, *par):
            ym = self._model(par)

            if self.upper_mask is None:
                result = -np.sum(self.logpdf(self.y, ym, self.yerror))
            else:
                # Measurements
                result = -np.sum(
                    self.logpdf(self.y[~self.upper_mask], ym[~self.upper_mask], self.yerror[~self.upper_mask])
                )
                # Upper limits, Tobit model
                # https://stats.stackexchange.com/questions/49443/how-to-model-this-odd-shaped-distribution-almost-a-reverse-j
                result += -np.sum(
                    self.logcdf((self.y[self.upper_mask] - ym[self.upper_mask]) / self.yerror[self.upper_mask])
                )

            # Barriers around parameter ranges (see _BARRIER_STRENGTH for the coefficient).
            result += self._BARRIER_STRENGTH * np.sum(self.barrier((par - self.limits0) / self.limits_scale))
            result += self._BARRIER_STRENGTH * np.sum(self.barrier((self.limits1 - par) / self.limits_scale))

            if self._prior is not None:
                idx, mean, inv_sigma2 = self._prior
                # Gaussian prior contributes -ln N = 0.5·((p-μ)/σ)² to the NLL.
                result += 0.5 * np.sum(inv_sigma2 * (np.asarray(par)[idx] - mean) ** 2)

            return result

        def grad(self, *par):
            """Analytic gradient of the cost. Requires `jac` to have been provided."""
            if self.jac is None:
                raise RuntimeError("MaximumLikelihood.grad called without a Jacobian")
            if self.upper_mask is not None:
                raise NotImplementedError("Analytic gradient not implemented with upper_mask")

            ym = self._model(par)
            j = self.jac(self.x, *par)  # shape (n_params, n_obs)

            residual = self.y - ym
            # d/dθ_k [(1/2) Σ ((y - m)/σ)²] = -Σ (y - m)/σ² · ∂m/∂θ_k
            g = -(j @ (residual * self._inv_yerror2))

            par_arr = np.asarray(par)
            # d barrier((p-lo)/s)/dp = -s/(p-lo)²;  d barrier((hi-p)/s)/dp = s/(hi-p)²
            g += (
                self._BARRIER_STRENGTH
                * self.limits_scale
                * (1.0 / (self.limits1 - par_arr) ** 2 - 1.0 / (par_arr - self.limits0) ** 2)
            )

            if self._prior is not None:
                idx, mean, inv_sigma2 = self._prior
                # d/dp [0.5·inv_sigma2·(p-μ)²] = inv_sigma2·(p-μ)
                g[idx] += inv_sigma2 * (par_arr[idx] - mean)
            return g

        @staticmethod
        def logpdf(x, mu, sigma):
            # We do not need the second term as it does not depend on parameters
            return -(((x - mu) / sigma) ** 2) / 2  # - np.log(np.sqrt(2*np.pi) * sigma)

        @staticmethod
        def barrier(x):
            res = np.where(x > 0, 1 / x, np.inf)  # FIXME: naive barrier function

            return res

        @staticmethod
        def logcdf(x):
            # TODO: faster (maybe not so accurate, as we do not need it) implementation
            # return norm.logcdf(x)

            result = np.zeros(len(x))

            idx = x < -5
            result[idx] = -(x[idx] ** 2) / 2 - 1 / x[idx] ** 2 - 0.9189385336 - np.log(-x[idx])
            result[~idx] = np.log(0.5) + np.log1p(erf(x[~idx] / np.sqrt(2)))

            return result


__all__ = ["MaximumLikelihood"]
