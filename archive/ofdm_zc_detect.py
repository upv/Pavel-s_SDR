"""
Q15 OFDM waveform analysis
==========================
Q1: Read interleaved Q15 (int16) IQ file -> complex float in [-1, +1)
Q2: Downsample 7.68 MHz -> 1.92 MHz (decimate by 4, anti-alias FIR)
Q3: Blind Zadoff-Chu pilot search over roots {21, 23, 25}:
    - which OFDM symbols carry the ZC pilot (expected: n and n+2)
    - most likely root and starting sample
    - correlation plots with detected peaks marked

Signal parameters (at baseband rate 1.92 MHz):
    NFFT = 128, CP = 16 -> symbol length 144, 9 symbols total
    Pilots: ZC length 127, DC bin nulled
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
FS_IN      = 7.68e6          # capture rate
OSR        = 4               # oversampling ratio
FS_BB      = FS_IN / OSR     # 1.92 MHz baseband rate
NFFT       = 128
CP         = 16
SYM_LEN    = NFFT + CP       # 144 samples @ 1.92 MHz
N_SYMBOLS  = 9
ZC_LEN     = 127
ROOTS      = [21, 23, 25]

# ----------------------------------------------------------------------
# Q1: read Q15 interleaved IQ -> complex float in [-1, +1)
# ----------------------------------------------------------------------
def read_q15_iq(path):
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2:
        raw = raw[:-1]                       # guard against odd tail
    i = raw[0::2].astype(np.float64) / 32768.0   # Q15: divide by 2^15
    q = raw[1::2].astype(np.float64) / 32768.0
    return i + 1j * q

# ----------------------------------------------------------------------
# Q2: decimate by 4 with anti-aliasing filter
# ----------------------------------------------------------------------
def downsample_x4(x):
    # Linear-phase FIR lowpass, cutoff at 0.8 * (Fs_out/2) to leave a
    # transition band, then polyphase decimation. zero_phase=True removes
    # the filter group delay so sample indices stay aligned.
    return signal.decimate(x, OSR, ftype="fir", n=127, zero_phase=True)

# ----------------------------------------------------------------------
# Reference pilot generation (matches the transmitter description)
# ----------------------------------------------------------------------
def zc_sequence(root, n=ZC_LEN):
    m = np.arange(n)
    return np.exp(-1j * np.pi * root * m * (m + 1) / n)

def pilot_time_symbol(root, centered=True):
    """Build the time-domain OFDM pilot symbol (without CP).

    Subcarrier map per the spec: 128 bins, DC at bin index 64 (0-based),
    lower side = bins 0..63, upper side = bins 65..127, ZC split across
    them in order. `centered=True` treats that layout as an fftshifted
    spectrum (ifftshift before the IFFT); `centered=False` treats the bin
    indices as raw FFT indices. Both are tried during detection since the
    transmitter's convention isn't fully pinned down by the comment.
    """
    zc = zc_sequence(root)
    X = np.zeros(NFFT, dtype=complex)
    X[0:64]   = zc[0:64]
    X[64]     = 0.0                  # DC null
    X[65:128] = zc[64:127]
    if centered:
        X = np.fft.ifftshift(X)
    x = np.fft.ifft(X) * np.sqrt(NFFT)   # unit-ish power reference
    return x

# ----------------------------------------------------------------------
# Q3: blind search over roots and time offsets
# ----------------------------------------------------------------------
def normalized_xcorr(x, ref):
    """|cross-correlation| normalized by local signal energy -> [0, 1]."""
    L = len(ref)
    num = np.abs(signal.correlate(x, ref, mode="valid"))
    energy = signal.correlate(np.abs(x) ** 2, np.ones(L), mode="valid").real
    den = np.sqrt(np.maximum(energy, 1e-12) * np.sum(np.abs(ref) ** 2))
    return num / den

def search_zc(x_bb):
    results = {}
    for root in ROOTS:
        best = None
        for centered in (True, False):
            ref = pilot_time_symbol(root, centered)
            c = normalized_xcorr(x_bb, ref)
            k = int(np.argmax(c))
            if best is None or c[k] > best["peak"]:
                best = {"corr": c, "idx": k, "peak": float(c[k]),
                        "centered": centered}
        results[root] = best
    return results

def estimate_signal_onset(x, win=CP):
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
    n_pilot = int(np.clip(round((p1 - CP - onset) / SYM_LEN),
                          0, N_SYMBOLS - 3))       # n+2 must be <= last symbol
    frame_start = p1 - CP - n_pilot * SYM_LEN

    return {
        "root": best_root,
        "peak_val": r["peak"],
        "pilot1_pos": p1,
        "pilot2_pos": p2,
        "pilot_symbols": (n_pilot, n_pilot + 2),
        "frame_start": frame_start,
        "centered": r["centered"],
    }

# ----------------------------------------------------------------------
def main(path="q15_ofdm_waveform.bin"):
    # Q1
    x = read_q15_iq(path)
    print(f"Q1: read {len(x)} complex samples @ {FS_IN/1e6:.2f} Msps, "
          f"range [{min(x.real.min(), x.imag.min()):+.4f}, "
          f"{max(x.real.max(), x.imag.max()):+.4f}]")

    # Q2 — but only if the capture is genuinely oversampled. A 4x-oversampled
    # waveform carries ~all its energy in the center 1/4 of the band; this
    # file measures ~25% there (flat spectrum), i.e. it was generated at
    # FFT128/CP16 directly at the file rate, so decimation would destroy it.
    X = np.fft.fftshift(np.fft.fft(x * np.hanning(len(x))))
    P = np.abs(X) ** 2
    frac_center = P[3*len(P)//8:5*len(P)//8].sum() / P.sum()
    if frac_center > 0.8:
        x_bb = downsample_x4(x)
        print(f"Q2: decimated to {len(x_bb)} samples @ {FS_BB/1e6:.2f} Msps "
              f"(center-band energy {frac_center*100:.0f}%)")
    else:
        x_bb = x
        print(f"Q2: spectrum is full-band (center-band energy only "
              f"{frac_center*100:.0f}%) -> file is already at symbol rate; "
              f"decimation skipped")

    # Q3
    results = search_zc(x_bb)
    print("\nQ3: peak normalized correlation per candidate root:")
    for root in ROOTS:
        r = results[root]
        print(f"    root {root}: peak = {r['peak']:.3f} at sample {r['idx']}"
              f"  (spectrum layout: {'centered' if r['centered'] else 'raw'})")

    det = interpret(results, x_bb)
    print(f"\n  -> detected root index : {det['root']}")
    print(f"  -> pilot #1 FFT-part starts at sample {det['pilot1_pos']} "
          f"(@1.92 MHz)")
    if det["pilot2_pos"] is not None:
        print(f"  -> pilot #2 confirmed at sample {det['pilot2_pos']} "
              f"(= pilot#1 + 2 symbols)")
    n1, n2 = det["pilot_symbols"]
    print(f"  -> ZC pilots occupy OFDM symbols {n1} and {n2} (0-based)")
    print(f"  -> estimated frame start (incl. CP): sample "
          f"{det['frame_start']} @1.92 MHz "
          f"(= {det['frame_start']*OSR} @7.68 MHz)")

    # Plots
    fig, axes = plt.subplots(len(ROOTS), 1, figsize=(11, 8), sharex=True)
    for ax, root in zip(axes, ROOTS):
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
    print("\nSaved plot: zc_correlation.png")
    return det

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "q15_ofdm_waveform.bin")
