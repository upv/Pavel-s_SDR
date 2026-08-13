#!/usr/bin/env python3
"""
Task 1 — Q15 OFDM waveform: convert, downsample, and detect Zadoff-Chu pilots.

Signal parameters
    FFT size              128
    Cyclic prefix         16
    OFDM symbols          9
    Native rate           1.92 MHz  (LTE 1.4 MHz numerology)
    File rate (stated)    7.68 MHz  (4x oversampled)
    Pilot roots           21, 23, 25
    Pilot symbols         n and n+2
    Data                  QPSK
    ZC length             127 (DC skipped)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
BIN_PATH = Path(__file__).resolve().parent / "q15_ofdm_waveform.bin"

N_FFT = 128
CP_LEN = 16
SYM_LEN = N_FFT + CP_LEN  # 144 samples at 1.92 MHz
N_SYM = 9
ZC_LEN = 127
PILOT_ROOTS = (21, 23, 25)

FS_CAPTURE = 7.68e6  # Hz, as stated in the assignment
FS_NATIVE = 1.92e6  # Hz, OFDM native rate (decimate by 4)
DECIM = 4
DC_BIN = N_FFT // 2  # bin 64 in a 128-point fftshifted grid (MATLAB index 65)


# ---------------------------------------------------------------------------
# Q1 — Read Q15 interleaved IQ and convert to complex float in [-1, +1)
# ---------------------------------------------------------------------------
def read_q15_iq(path: Path) -> np.ndarray:
    """Read interleaved int16 I/Q and scale by 2^-15."""
    raw = np.fromfile(path, dtype=np.int16)
    if raw.size % 2:
        raise ValueError(f"{path} has an odd number of int16 values")
    iq = raw.astype(np.float64) / 32768.0  # Q15: [-1, +1)
    return iq[0::2] + 1j * iq[1::2]


# ---------------------------------------------------------------------------
# Q2 — Anti-alias and downsample 7.68 MHz -> 1.92 MHz
# ---------------------------------------------------------------------------
def downsample_4x(x: np.ndarray, fs_hi: float = FS_CAPTURE) -> np.ndarray:
    """FIR low-pass (cutoff just below the 1.92 MHz Nyquist) then keep every 4th sample."""
    cutoff = 0.9 * (fs_hi / DECIM) / 2.0  # 0.9 * 960 kHz
    taps = signal.firwin(numtaps=65, cutoff=cutoff, fs=fs_hi)
    return signal.lfilter(taps, 1.0, x)[::DECIM]


# ---------------------------------------------------------------------------
# Zadoff-Chu local replicas
# ---------------------------------------------------------------------------
def zadoff_chu(root: int, length: int = ZC_LEN) -> np.ndarray:
    """seq[m] = exp(-j * pi * R * m * (m+1) / N), m = 0..N-1."""
    m = np.arange(length, dtype=np.float64)
    return np.exp(-1j * np.pi * root * m * (m + 1.0) / length)


def map_zc_to_fft_grid(zc: np.ndarray) -> np.ndarray:
    """
    Place a length-127 ZC onto 128 FFT bins in fftshift order:
        bins 0..63   negative frequencies (incl. Nyquist)
        bin  64      DC  (left unused)
        bins 65..127 positive frequencies
    """
    grid = np.zeros(N_FFT, dtype=np.complex128)
    grid[:DC_BIN] = zc[:DC_BIN]
    grid[DC_BIN] = 0.0
    grid[DC_BIN + 1 :] = zc[DC_BIN:]
    return grid


def ofdm_pilot_time(root: int, with_cp: bool = True) -> np.ndarray:
    """IFFT of the mapped ZC, optionally with a cyclic prefix."""
    grid = map_zc_to_fft_grid(zadoff_chu(root))
    time_sym = np.fft.ifft(np.fft.ifftshift(grid), n=N_FFT)
    if with_cp:
        return np.concatenate([time_sym[-CP_LEN:], time_sym])
    return time_sym


def extract_non_dc(X_shifted: np.ndarray) -> np.ndarray:
    return np.concatenate([X_shifted[:DC_BIN], X_shifted[DC_BIN + 1 :]])


# ---------------------------------------------------------------------------
# Q3 — Blind search: unknown root, unknown start
# ---------------------------------------------------------------------------
def normalized_xcorr(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """|matched filter| / sqrt(window energy * ||ref||^2), length = len(x)-len(ref)+1."""
    corr = signal.correlate(x, ref, mode="valid")
    energy = signal.correlate(np.abs(x) ** 2, np.ones(len(ref)), mode="valid")
    denom = np.sqrt(np.maximum(energy * np.vdot(ref, ref).real, 1e-30))
    return np.abs(corr) / denom


def search_zc(rx: np.ndarray):
    """Correlate against CP-prefixed OFDM pilot replicas for every candidate root."""
    corrs = {}
    best = {"peak": -1.0, "root": None, "start": None}
    for root in PILOT_ROOTS:
        ref = ofdm_pilot_time(root, with_cp=True)
        mag = normalized_xcorr(rx, ref)
        peak_idx = int(np.argmax(mag))
        peak_val = float(mag[peak_idx])
        corrs[root] = mag
        if peak_val > best["peak"]:
            best = {"peak": peak_val, "root": root, "start": peak_idx, "ref_len": len(ref)}
    return corrs, best


def per_symbol_zc_scores(rx: np.ndarray, timing: int = 0) -> np.ndarray:
    """
    For each of 9 OFDM symbols, correlate the 127 used subcarriers with each root.
    Returns array shape (N_SYM, n_roots).
    """
    scores = np.zeros((N_SYM, len(PILOT_ROOTS)))
    for s in range(N_SYM):
        start = timing + s * SYM_LEN + CP_LEN
        useful = rx[start : start + N_FFT]
        if useful.size < N_FFT:
            continue
        X = np.fft.fftshift(np.fft.fft(useful, n=N_FFT))
        sc = extract_non_dc(X)
        sc_n = np.linalg.norm(sc)
        for i, root in enumerate(PILOT_ROOTS):
            zc = zadoff_chu(root)
            scores[s, i] = np.abs(np.vdot(zc, sc)) / (np.linalg.norm(zc) * sc_n + 1e-30)
    return scores


def find_pilot_symbol_indices(scores: np.ndarray, thresh: float = 0.8) -> list[int]:
    best_root_col = int(np.argmax(scores.max(axis=0)))
    hits = np.where(scores[:, best_root_col] > thresh)[0].tolist()
    return hits, best_root_col


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_q1_q2(x_q15: np.ndarray, x_ds: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    t = np.arange(len(x_q15)) / FS_CAPTURE * 1e6
    axes[0, 0].plot(t, x_q15.real, lw=0.6, label="I")
    axes[0, 0].plot(t, x_q15.imag, lw=0.6, label="Q")
    axes[0, 0].set_title("Q1: Q15 → complex float")
    axes[0, 0].set_xlabel("Time (µs)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    f_hi, psd_hi = signal.welch(
        x_q15, fs=FS_CAPTURE, nperseg=min(256, len(x_q15)), return_onesided=False
    )
    order = np.argsort(f_hi)
    axes[0, 1].plot(f_hi[order] / 1e6, 10 * np.log10(psd_hi[order] + 1e-18))
    axes[0, 1].set_title("Q1: spectrum at 7.68 MHz (as labeled)")
    axes[0, 1].set_xlabel("Frequency (MHz)")
    axes[0, 1].set_ylabel("PSD (dB)")
    axes[0, 1].grid(True, alpha=0.3)

    t_ds = np.arange(len(x_ds)) / FS_NATIVE * 1e6
    axes[1, 0].plot(t_ds, x_ds.real, lw=0.6, label="I")
    axes[1, 0].plot(t_ds, x_ds.imag, lw=0.6, label="Q")
    axes[1, 0].set_title("Q2: after anti-alias + decimate ×4")
    axes[1, 0].set_xlabel("Time (µs)")
    axes[1, 0].set_ylabel("Amplitude")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    nfft = min(256, len(x_ds))
    f_lo, psd_lo = signal.welch(x_ds, fs=FS_NATIVE, nperseg=nfft, return_onesided=False)
    order = np.argsort(f_lo)
    axes[1, 1].plot(f_lo[order] / 1e6, 10 * np.log10(psd_lo[order] + 1e-18))
    axes[1, 1].set_title("Q2: spectrum at 1.92 MHz")
    axes[1, 1].set_xlabel("Frequency (MHz)")
    axes[1, 1].set_ylabel("PSD (dB)")
    axes[1, 1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "q1_q2_iq_downsample.png", dpi=120)


def plot_q3(corrs: dict, best: dict, scores: np.ndarray, pilot_syms: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    for root, mag in corrs.items():
        ax.plot(mag, lw=1.1, label=f"root {root}")
    pk = best["start"]
    ax.plot(pk, best["peak"], "r*", ms=14, zorder=5, label=f"peak @ {pk}")
    ax.axvline(pk, color="r", ls="--", lw=0.8, alpha=0.7)
    # second pilot is 2 symbols later (hint: n and n+2)
    pk2 = pk + 2 * SYM_LEN
    mag_best = corrs[best["root"]]
    if pk2 < len(mag_best):
        ax.plot(pk2, mag_best[pk2], "r*", ms=10, zorder=5)
        ax.annotate("n+2", xy=(pk2, mag_best[pk2]), xytext=(pk2 + 20, 0.85))
    ax.set_title("Q3.3: time-domain correlation vs lag")
    ax.set_xlabel("Starting sample (1.92 MHz)")
    ax.set_ylabel("Normalized |correlation|")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    ax = axes[1]
    x = np.arange(N_SYM)
    width = 0.25
    for i, root in enumerate(PILOT_ROOTS):
        ax.bar(x + (i - 1) * width, scores[:, i], width=width, label=f"root {root}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in x])
    ax.set_xlabel("OFDM symbol index (0-based)")
    ax.set_ylabel("Freq-domain |⟨ZC, subcarriers⟩|")
    ax.set_title("Q3.1: per-symbol Zadoff-Chu score")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)
    for s in pilot_syms:
        ax.annotate("pilot", xy=(s, 1.0), xytext=(s, 1.08), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "q3_zc_correlation.png", dpi=120)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Q1
    x = read_q15_iq(BIN_PATH)
    print("=" * 64)
    print("Q1  Read and convert Q15 IQ")
    print("=" * 64)
    print(f"  file          : {BIN_PATH.name}")
    print(f"  complex samples: {len(x)}")
    print(f"  I range       : [{x.real.min():+.6f}, {x.real.max():+.6f}]")
    print(f"  Q range       : [{x.imag.min():+.6f}, {x.imag.max():+.6f}]")
    print(f"  |z| max       : {np.abs(x).max():.6f}  (must be < 1)")
    print("  scaling       : int16 / 32768  →  [-1, +1)")

    # Q2
    x_ds = downsample_4x(x)
    print()
    print("=" * 64)
    print("Q2  Downsample 7.68 MHz → 1.92 MHz")
    print("=" * 64)
    print(f"  FIR cutoff    : {0.9 * (FS_CAPTURE / DECIM) / 2 / 1e3:.1f} kHz, decimate by {DECIM}")
    print(f"  length after  : {len(x_ds)} samples")
    print(f"  9 symbols @ 1.92 MHz would be {N_SYM * SYM_LEN} samples")
    if len(x) == N_SYM * SYM_LEN:
        print("  NOTE: the file is already 1296 samples = 9 × (128+16),")
        print("        i.e. native 1.92 MHz. ZC search uses this signal,")
        print("        not the extra ×4 decimation (that would leave 324 samples).")
        rx = x
        fs = FS_NATIVE
    else:
        rx = x_ds
        fs = FS_NATIVE

    # Q3
    corrs, best = search_zc(rx)
    scores = per_symbol_zc_scores(rx, timing=0)
    pilot_syms, root_col = find_pilot_symbol_indices(scores)
    detected_root = PILOT_ROOTS[root_col]

    # If the frame is not aligned at sample 0, infer timing from the matched-filter peak
    frame_start = best["start"] % SYM_LEN
    scores_aligned = per_symbol_zc_scores(rx, timing=frame_start)
    pilot_syms_aligned, _ = find_pilot_symbol_indices(scores_aligned)

    print()
    print("=" * 64)
    print("Q3  Zadoff-Chu search")
    print("=" * 64)
    print("  3.2 most likely root index :", best["root"])
    print("  3.2 starting position      : sample", best["start"],
          f"({best['start'] / fs * 1e6:.3f} µs at {fs/1e6:.2f} MHz)")
    print(f"       (start of CP of the first pilot OFDM symbol; "
          f"useful IFFT starts at {best['start'] + CP_LEN})")
    print("  3.2 peak correlation       :", f"{best['peak']:.6f}")
    print()
    print("  3.1 Zadoff-Chu OFDM symbols (0-based):", pilot_syms_aligned)
    if len(pilot_syms_aligned) >= 2:
        n = pilot_syms_aligned[0]
        print(f"       n = {n}, n+2 = {n + 2}  "
              f"(MATLAB 1-based: symbols {n + 1} and {n + 3})")
    print()
    print("  Per-symbol freq-domain scores (timing = "
          f"{frame_start}):")
    hdr = "    sym | " + "  ".join(f"root {r:>2d}" for r in PILOT_ROOTS)
    print(hdr)
    for s in range(N_SYM):
        row = "    %3d |" % s
        for i in range(len(PILOT_ROOTS)):
            row += f"   {scores_aligned[s, i]:6.3f}"
        if s in pilot_syms_aligned:
            row += "   <-- ZC pilot"
        print(row)

    plot_q1_q2(x, x_ds)
    plot_q3(corrs, best, scores_aligned, pilot_syms_aligned)
    print()
    print("Saved q1_q2_iq_downsample.png and q3_zc_correlation.png")
    plt.show()


if __name__ == "__main__":
    main()
