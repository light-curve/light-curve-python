import pytest


@pytest.fixture(scope="session")
def atcat_lsst_band_groups():
    return {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4, "Y": 5}
