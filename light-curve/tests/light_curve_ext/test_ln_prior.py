import copy
import pickle

import pytest

from light_curve.light_curve_ext import ln_prior

LN_PRIORS = [
    ln_prior.none(),
    ln_prior.log_normal(1.0, 2.0),
    ln_prior.log_uniform(1.0, 10.0),
    ln_prior.normal(-1.0, 3.0),
    ln_prior.uniform(-2.0, 1.0),
    ln_prior.mix([(0.5, ln_prior.uniform(0.0, 1.0)), (0.5, ln_prior.normal(0.5, 0.1))]),
]


@pytest.mark.parametrize("lnpr", LN_PRIORS)
@pytest.mark.parametrize("pickle_protocol", tuple(range(2, pickle.HIGHEST_PROTOCOL + 1)))
def test_pickle(lnpr, pickle_protocol):
    b = pickle.dumps(lnpr, protocol=pickle_protocol)
    pickle.loads(b)


@pytest.mark.parametrize("lnpr", LN_PRIORS)
def test_copy(lnpr):
    copy.copy(lnpr)


@pytest.mark.parametrize("lnpr", LN_PRIORS)
def test_deepcopy(lnpr):
    copy.deepcopy(lnpr)
