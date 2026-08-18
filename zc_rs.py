import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

path = './q15_ofdm_waveform.bin'

N_FFT = 128
CP_LEN = 16
SYM_LEN = N_FFT + CP_LEN  # 144 samples at 1.92 MHz
N_SYM = 9
ZC_LEN = 127
PILOT_ROOTS = (21, 23, 25)

FS_CAPTURE = 7.68e6  # Hz, as stated in the assignment
FS_NATIVE = 1.92e6  # Hz, OFDM native rate (decimate by 4)
DECIM = 4
DC_BIN = N_FFT // 2  # bin 64 in a 128-point fftshifted grid

# Zadoff-Chu local replicas
# Generates Zadoff-Chu squences for given roots and length of sequence
def zc_sequence(root, n=ZC_LEN):
    m = np.arange(n)
    return np.exp(-1j * np.pi * root * m * (m + 1) / n)


def pilot_time_symbol(root):
    """
    Subcarrier map per the spec: 128 bins, DC at bin index 64 (0-based),
    lower side = bins 0..63, upper side = bins 65..127, ZC split across
    them in order
    """
    zc = zc_sequence(root)
    X = np.zeros(N_FFT, dtype=complex)
    X[0:64]   = zc[0:64]
    X[64]     = 0.0                  # DC null
    X[65:128] = zc[64:127]
    
    X = np.fft.ifftshift(X)

    x = np.fft.ifft(X) * np.sqrt(N_FFT)   # unit-ish power reference
    return x

def normalized_xcorr(x, ref):
    """|cross-correlation| normalized by local signal energy -> [0, 1]."""
    L = len(ref)
    num = np.abs(signal.correlate(x, ref, mode="valid"))
    energy = signal.correlate(np.abs(x) ** 2, np.ones(L), mode="valid").real
    den = np.sqrt(np.maximum(energy, 1e-12) * np.sum(np.abs(ref) ** 2))
    return num / den

def search_zc(x_bb):
    results = {}
    for root in PILOT_ROOTS:
        best = None
        
        ref = pilot_time_symbol(root)
        c = normalized_xcorr(x_bb, ref)
        k = int(np.argmax(c))
        if best is None or c[k] > best["peak"]:
            best = {"corr": c, "idx": k, "peak": float(c[k])}
            
        results[root] = best
    return results

def estimate_signal_onset(x, win=CP_LEN):
    """First sample where moving-average power rises well above the noise
    floor (noise floor taken from the quietest 10% of windows)."""
    p = np.convolve(np.abs(x) ** 2, np.ones(win) / win, mode="valid")
    floor = np.percentile(p, 10)
    thr = max(10 * floor, 0.05 * p.max())
    idx = np.flatnonzero(p > thr)
    return int(idx[0]) if idx.size else 0

def interpret(results, x_bb):
    best_root = max(results, key=lambda r: results[r]["peak"])
    r = results[best_root]
    c = r["corr"]

    # First pilot peak, then look for the partner exactly 2 symbols away
    p1 = r["idx"]
    lo, hi = p1 + 2 * SYM_LEN - 2, p1 + 2 * SYM_LEN + 3
    p2 = None
    if hi < len(c):
        w = c[lo:hi]
        if w.max() > 0.5 * r["peak"]:
            p2 = lo + int(np.argmax(w))

    # The correlation peaks at the FFT-part start of pilot symbol n, i.e.
    # frame_start + n*SYM_LEN + CP. A single pilot leaves n ambiguous, so
    # anchor the frame start with an energy-onset estimate, snap it to the
    # nearest consistent symbol grid position, and read n off the grid.
    onset = estimate_signal_onset(x_bb)
    n_pilot = int(np.clip(round((p1 - CP_LEN - onset) / SYM_LEN),
                          0, N_SYM - 3))       # n+2 must be <= last symbol
    frame_start = p1 - CP_LEN - n_pilot * SYM_LEN

    return {
        "root": best_root,
        "peak_val": r["peak"],
        "pilot1_pos": p1,
        "pilot2_pos": p2,
        "pilot_symbols": (n_pilot, n_pilot + 2),
        "frame_start": frame_start
    }


raw = np.fromfile(path, dtype=np.int16)

iq = raw.astype(np.float64) / 32768.0  # Q15: [-1, +1)
iq_comp = iq[0::2] + 1j * iq[1::2]

X = np.fft.fftshift(np.fft.fft(iq_comp * np.hanning(len(iq_comp))))
P = np.abs(X) ** 2
frac_center = P[3*len(P)//8:5*len(P)//8].sum() / P.sum()
if frac_center > 0.8:
    cutoff = 0.9 * (FS_CAPTURE / DECIM) / 2.0  # 0.9 * 960 kHz
    taps = signal.firwin(numtaps=65, cutoff=cutoff, fs=FS_CAPTURE)
    sig_filt = signal.lfilter(taps, 1.0, iq_comp)[::DECIM]
else:
    sig_filt = iq_comp



results = search_zc(sig_filt)

det = interpret(results, sig_filt)

# Plots
fig, axes = plt.subplots(len(PILOT_ROOTS), 1, figsize=(11, 8), sharex=True)
for ax, root in zip(axes, PILOT_ROOTS):
    c = results[root]["corr"]
    ax.plot(c, lw=0.7)
    ax.set_ylabel(f"root {root}")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    if root == det["root"]:
        ax.plot(det["pilot1_pos"], c[det["pilot1_pos"]], "rv", ms=10,
                label=f"pilot 1 @ {det['pilot1_pos']}")
        if det["pilot2_pos"] is not None:
            ax.plot(det["pilot2_pos"], c[det["pilot2_pos"]], "g^", ms=10,
                    label=f"pilot 2 @ {det['pilot2_pos']}")
        ax.legend(loc="upper right")
        ax.set_title(f"root {root}  <-- DETECTED "
                        f"(peak {det['peak_val']:.3f})", color="crimson")
axes[-1].set_xlabel("sample index @ 1.92 MHz")
fig.suptitle("Normalized cross-correlation with candidate ZC pilot symbols")
fig.tight_layout()
fig.savefig("zc_correlation.png", dpi=130)












