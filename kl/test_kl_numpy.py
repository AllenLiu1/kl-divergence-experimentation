from . import kl_numpy
import scipy.stats
import numpy as np

def test_kl():
    p = [0.1, 0.5, 0.3, 0.1]
    q = [0.2, 0.2, 0.3, 0.3]
    res = np.round(kl_numpy.kl(p, q),5)
    res_scipy = np.round(scipy.stats.entropy(p, q),5)
    assert res == res_scipy, f"res={res} != res_scipy={res_scipy}"

