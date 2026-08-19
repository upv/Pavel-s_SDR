
"""
Task 2: STO and CFO estimation for impaired_waveform_iq_float32.bin
===================================================================
Model:  r(k) = s(k - theta) * exp(j*2*pi*eps*k/N) + n(k)
        N = 1024 (FFT), CP = 128, 6 OFDM symbols, float32 interleaved IQ.

Method: CP-based ML estimation

Q4:  theta_hat = argmax_d { |gamma(d)| - rho*Phi(d) }

Q5:  In the CP region  r(k)*conj(r(k+N)) = |s|^2 * exp(-j*2*pi*eps),
     so  eps_hat = -angle(gamma(theta_hat)) / (2*pi)   [subcarrier units]
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = "./impaired_waveform_iq_float32.bin"

N       = 1024          # FFT size (the "N=128" in the model text is a typo;
CP      = 128           # the file spec says FFT 1024 / CP 128)
SYM_LEN = N + CP        # 1152
N_SYM   = 6

def read_f32_iq(path):
    raw = np.fromfile(path, dtype=np.float32).astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]

def cp_metric(r):
    """gamma(d), Phi(d) for every candidate offset d in the file."""
    L = CP
    #   Simplified vesrion of cross-correlation for defined window-size
    prod = r[:-N] * np.conj(r[N:])                    # r(k) r*(k+N)
    pwr  = 0.5 * (np.abs(r[:-N])**2 + np.abs(r[N:])**2)
    w = np.ones(L)
    gamma = np.convolve(prod, w)[L-1:len(prod)]       # calculate sliding sums
    phi   = np.convolve(pwr,  w)[L-1:len(pwr)]
    return gamma, phi

def estimate(r, rho=0.99):
    gamma, phi = cp_metric(r)
    metric = np.abs(gamma) - rho * phi #    Metric should be refined for energy, 
                                        #   rho = SNR / (SNR +1) - for Gaussian model
                                        # phi - average energy. ML metric

    # Fold onto one symbol period and average across symbols: the CP
    # correlation peak repeats every SYM_LEN samples.
    n_fold = len(metric) // SYM_LEN
    folded_metric = metric[:n_fold*SYM_LEN].reshape(n_fold, SYM_LEN).sum(0)
    folded_gamma  = gamma[:n_fold*SYM_LEN].reshape(n_fold, SYM_LEN).sum(0)

    theta = int(np.argmax(folded_metric))
    eps   = -np.angle(folded_gamma[theta]) / (2*np.pi)
    return theta, eps, gamma, folded_metric, folded_gamma


#   read samples
r = read_f32_iq(path)

#   Estimate required parameteres: Theta - Symbol Timing Offset
#   eps - Carrier Frequency Offset, gamma - sliding window cross-correlation, 
#   folded - folded metric for 6 symbols, fgamma - folded gamma for 6 symbols
theta, eps, gamma, folded, fgamma = estimate(r)

print(f"\nQ4: STO  theta_hat = {theta} samples")
print(f"Q5: CFO  eps_hat   = {eps:+.5f} subcarrier spacings "
        f"(fractional part; = {eps:+.5f} * fs/{N} Hz for sample rate fs)")

# Per-symbol CFO estimates as a consistency check
per_sym = []
for m in range((len(r) - N - CP) // SYM_LEN + 1):
    d = theta + m * SYM_LEN
    if d + CP + N <= len(r):
        g = np.sum(r[d:d+CP] * np.conj(r[d+N:d+N+CP]))
        per_sym.append(-np.angle(g) / (2*np.pi))
print(f"    per-symbol eps: {np.array2string(np.array(per_sym), precision=5)}")
print(f"    std across symbols: {np.std(per_sym):.2e}")

fig, axes = plt.subplots(2, 1, figsize=(11, 7))
axes[0].plot(np.abs(gamma), lw=0.8)
for m in range(N_SYM):
    d = theta + m * SYM_LEN
    if d < len(gamma):
        axes[0].axvline(d, color="crimson", ls="--", alpha=0.6)
axes[0].set_title("|gamma(d)| — CP correlation over the whole file "
                    "(dashed: detected symbol starts)")
axes[0].set_xlabel("offset d (samples)"); axes[0].grid(alpha=0.3)

axes[1].plot(folded, lw=1.0)
axes[1].plot(theta, folded[theta], "rv", ms=11,
                label=f"theta_hat = {theta}")
axes[1].set_title("Folded ML metric  sum_m |gamma| - rho*Phi  "
                    "(one symbol period, averaged over symbols)")
axes[1].set_xlabel("offset within symbol period"); axes[1].grid(alpha=0.3)
axes[1].legend()
fig.tight_layout()
out = Path(__file__).resolve().parent / "task2_sto_cfo.png"
fig.savefig(out, dpi=130)
print(f"\nplot saved: {out.name}")









