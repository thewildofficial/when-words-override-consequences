"""Numpy check of the swap algebra used by coordinate_swap_fp32 (pre = h @ pinv(M).T; h' = h + (expected-pre) @ M.T)."""
import numpy as np
rng = np.random.default_rng(0)
d = 64

def swap(h, M, alpha):
    pre = h @ np.linalg.pinv(M).T
    exp = pre + alpha * (pre[..., ::-1] - pre)
    return h + (exp - pre) @ M.T

def test_same_basis_alpha1_recurrent_twice_is_identity():
    M = rng.normal(size=(d, 2)); h = rng.normal(size=(5, d))
    assert np.allclose(swap(swap(h, M, 1.0), M, 1.0), h, atol=1e-8)

def test_clean_delta_twice_does_not_cancel():
    M = rng.normal(size=(d, 2)); h = rng.normal(size=(5, d))
    delta = swap(h, M, 1.0) - h
    out = h + delta + delta  # two layers each add the clean-state delta
    assert not np.allclose(out, h, atol=1e-3)
    assert np.allclose(out - h, 2 * delta)

def test_alpha0_noop():
    M = rng.normal(size=(d, 2)); h = rng.normal(size=(3, d))
    assert np.allclose(swap(h, M, 0.0), h)
