"""CP-based STO/CFO estimator: synthetic checks plus the archived capture."""

import numpy as np
import pytest

import zc_part2 as m


def ofdm_frame(rng, n_sym=m.N_SYM):
    syms = []
    for _ in range(n_sym):
        X = (rng.choice([-1, 1], m.N) + 1j * rng.choice([-1, 1], m.N)) / np.sqrt(2)
        x = np.fft.ifft(X) * np.sqrt(m.N)
        syms.append(np.concatenate([x[-m.CP:], x]))
    return np.concatenate(syms)


@pytest.mark.parametrize("theta, eps", [(0, 0.1), (300, -0.3), (777, 0.42)])
def test_estimate_recovers_sto_and_cfo(theta, eps):
    rng = np.random.default_rng(theta)
    s = ofdm_frame(rng)
    # Delay by theta samples: the frame wraps so every candidate period has data.
    r = np.roll(s, theta)
    k = np.arange(len(r))
    r = r * np.exp(2j * np.pi * eps * k / m.N)
    r += 0.05 * (rng.standard_normal(len(r)) + 1j * rng.standard_normal(len(r)))

    theta_hat, eps_hat, *_ = m.estimate(r)

    assert theta_hat == theta
    assert eps_hat == pytest.approx(eps, abs=0.01)


def test_archived_capture():
    if not m.DEFAULT_PATH.exists():
        pytest.skip("archived capture not present")
    theta, eps, *_ = m.estimate(m.read_f32_iq(m.DEFAULT_PATH))
    assert theta == 0
    assert eps == pytest.approx(0.2504, abs=1e-3)


def test_main_runs_from_any_cwd(tmp_path, monkeypatch, capsys):
    if not m.DEFAULT_PATH.exists():
        pytest.skip("archived capture not present")
    monkeypatch.chdir(tmp_path)
    m.main([])
    assert "theta_hat = 0" in capsys.readouterr().out
