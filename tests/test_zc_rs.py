"""Zadoff-Chu pilot detection: synthetic checks plus the archived capture."""

import numpy as np
import pytest

import zc_rs as m


def qpsk_symbol(rng):
    X = np.zeros(m.N_FFT, dtype=complex)
    data = (rng.choice([-1, 1], m.ZC_LEN) + 1j * rng.choice([-1, 1], m.ZC_LEN)) / np.sqrt(2)
    X[:m.DC_BIN] = data[:m.DC_BIN]
    X[m.DC_BIN + 1:] = data[m.DC_BIN:]
    return np.fft.ifft(np.fft.ifftshift(X)) * np.sqrt(m.N_FFT)


def frame(rng, root, pilot_syms):
    out = []
    for s in range(m.N_SYM):
        x = m.pilot_time_symbol(root) if s in pilot_syms else qpsk_symbol(rng)
        out.append(np.concatenate([x[-m.CP_LEN:], x]))
    return np.concatenate(out)


def test_zc_roots_are_nearly_orthogonal():
    a, b = (m.zc_sequence(r) for r in m.PILOT_ROOTS[:2])
    assert abs(np.vdot(a, a)) == pytest.approx(m.ZC_LEN)
    assert abs(np.vdot(a, b)) < 0.15 * m.ZC_LEN


@pytest.mark.parametrize("root, pilots", [(21, [1, 3]), (23, [4, 6]), (25, [2, 4])])
def test_detects_root_and_pilot_symbols(root, pilots):
    rng = np.random.default_rng(root)
    rx = frame(rng, root, pilots)
    rx += 0.05 * (rng.standard_normal(len(rx)) + 1j * rng.standard_normal(len(rx)))

    corrs, best_root, best_start = m.search_zc(rx)
    frame_start = (best_start - m.CP_LEN) % m.SYM_LEN
    hits, col = m.find_pilot_symbol_indices(m.per_symbol_zc_scores(rx, frame_start))

    assert best_root == root
    assert frame_start == 0
    assert m.PILOT_ROOTS[col] == root
    assert hits == pilots


def test_archived_capture():
    if not m.DEFAULT_PATH.exists():
        pytest.skip("archived capture not present")
    rx = m.load_baseband(m.DEFAULT_PATH)
    _, best_root, best_start = m.search_zc(rx)
    frame_start = (best_start - m.CP_LEN) % m.SYM_LEN
    hits, col = m.find_pilot_symbol_indices(m.per_symbol_zc_scores(rx, frame_start))
    assert (best_root, best_start, frame_start) == (25, 304, 0)
    assert hits == [2, 4]
