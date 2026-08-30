"""
Q4/Q5: STO and CFO estimation for impaired_waveform_iq_float32.bin
==================================================================
Received model:
    r(k) = s(k - theta) * exp(j * 2 * pi * eps * k / N) + n(k)

Parameters from the file spec (the "N=128" in the formula text is a typo):
    N      = 1024   FFT size
    CP     = 128    cyclic prefix length
    N_SYM  = 6      OFDM symbols
    IQ     = float32 interleaved I Q I Q ...

Method: CP-based ML estimator (van de Beek et al., 1997).
The CP copies the last N samples of each OFDM symbol, so r(k) and r(k+N)
are correlated over a CP-length window at the true symbol start:

    gamma(d) = sum_{k=d}^{d+L-1} r(k) * conj(r(k+N))
    Phi(d)   = 0.5 * sum_{k=d}^{d+L-1} (|r(k)|^2 + |r(k+N)|^2)
    Lambda(d) = |gamma(d)| - rho * Phi(d)

Q4:  theta_hat = argmax_d Lambda(d)
     (metric is folded over one symbol period and summed across symbols)

Q5:  In the CP region, r(k)*conj(r(k+N)) = |s|^2 * exp(-j*2*pi*eps),
     so  eps_hat = -angle(gamma(theta_hat)) / (2*pi)
     (fractional CFO in units of subcarrier spacing; unambiguous for |eps|<0.5)
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN_PATH = Path(__file__).resolve().parent / "impaired_waveform_iq_float32.bin"

N       = 1024
CP      = 128
SYM_LEN = N + CP          # 1152
N_SYM   = 6


def read_f32_iq(path):
    raw = np.fromfile(path, dtype=np.float32).astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def cp_metric(r):
    """Sliding CP correlation gamma(d) and energy Phi(d)."""
    prod = r[:-N] * np.conj(r[N:])
    pwr  = 0.5 * (np.abs(r[:-N]) ** 2 + np.abs(r[N:]) ** 2)
    w = np.ones(CP)
    gamma = np.convolve(prod, w)[CP - 1 : len(prod)]
    phi   = np.convolve(pwr,  w)[CP - 1 : len(pwr)]
    return gamma, phi


def estimate(r, rho=0.99):
    """
    rho = SNR / (SNR + 1). 0.99 is a high-SNR default; the peak location
    is insensitive to the exact value as long as rho is close to 1.
    """
    gamma, phi = cp_metric(r)
    metric = np.abs(gamma) - rho * phi

    n_fold = len(metric) // SYM_LEN
    folded_metric = metric[: n_fold * SYM_LEN].reshape(n_fold, SYM_LEN).sum(0)
    folded_gamma  = gamma[: n_fold * SYM_LEN].reshape(n_fold, SYM_LEN).sum(0)

    theta = int(np.argmax(folded_metric))
    eps   = -np.angle(folded_gamma[theta]) / (2 * np.pi)
    return theta, eps, gamma, folded_metric, folded_gamma


def per_symbol_cfo(r, theta):
    """CFO from each symbol's CP, as a consistency check."""
    estimates = []
    for m in range(N_SYM):
        d = theta + m * SYM_LEN
        if d + CP + N > len(r):
            break
        g = np.sum(r[d : d + CP] * np.conj(r[d + N : d + N + CP]))
        estimates.append(-np.angle(g) / (2 * np.pi))
    return np.asarray(estimates)


def main(path=BIN_PATH):
    r = read_f32_iq(path)
    print(f"read {len(r)} complex samples = {len(r) / SYM_LEN:.3f} x {SYM_LEN} "
          f"(exactly {N_SYM} symbols expected)")

    theta, eps, gamma, folded, _ = estimate(r)
    per_sym = per_symbol_cfo(r, theta)

    print(f"\nQ4: STO  theta_hat = {theta} samples")
    print(f"Q5: CFO  eps_hat   = {eps:+.5f} subcarrier spacings "
          f"(fractional; unambiguous for |eps| < 0.5)")
    if per_sym.size:
        print(f"    per-symbol eps: {np.array2string(per_sym, precision=5)}")
        print(f"    std across symbols: {np.std(per_sym):.2e}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    axes[0].plot(np.abs(gamma), lw=0.8)
    for m in range(N_SYM):
        d = theta + m * SYM_LEN
        if d < len(gamma):
            axes[0].axvline(d, color="crimson", ls="--", alpha=0.6)
    axes[0].set_title("|gamma(d)| — CP correlation "
                      "(dashed: detected symbol starts)")
    axes[0].set_xlabel("offset d (samples)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(folded, lw=1.0)
    axes[1].plot(theta, folded[theta], "rv", ms=11, label=f"theta_hat = {theta}")
    axes[1].set_title("Folded ML metric  sum_m |gamma| - rho*Phi")
    axes[1].set_xlabel("offset within symbol period")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "sc_cfo_sto.png"
    fig.savefig(out, dpi=130)
    print(f"\nplot saved: {out.name}")


if __name__ == "__main__":
    main()
