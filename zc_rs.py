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

def extract_non_dc(X_shifted: np.ndarray) -> np.ndarray:
    return np.concatenate([X_shifted[:DC_BIN], X_shifted[DC_BIN + 1 :]])

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
    #   Calculate cross-correlation
    num = np.abs(signal.correlate(x, ref, mode="valid"))
    #   Calculate energy 
    energy = signal.correlate(np.abs(x) ** 2, np.ones(L), mode="valid").real
    # Calculate normalization to make correlation maximum equal to 1
    den = np.sqrt(np.maximum(energy, 1e-12) * np.sum(np.abs(ref) ** 2))
    return num / den

def search_zc(rx: np.ndarray):
    """Correlate against the OFDM pilot replica of every candidate root.

    Returns the per-root correlation magnitudes plus the root and sample index
    of the strongest peak.
    """
    corrs = {root: normalized_xcorr(rx, pilot_time_symbol(root))
             for root in PILOT_ROOTS}
    best_root = max(corrs, key=lambda r: corrs[r].max())
    best_start = int(np.argmax(corrs[best_root]))
    return corrs, best_root, best_start

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
            zc = zc_sequence(root)
            scores[s, i] = np.abs(np.vdot(zc, sc)) / (np.linalg.norm(zc) * sc_n + 1e-12)
    return scores

def find_pilot_symbol_indices(scores: np.ndarray,
                              thresh: float = 0.8) -> tuple[list[int], int]:
    best_root_col = int(np.argmax(scores.max(axis=0)))
    hits = np.where(scores[:, best_root_col] > thresh)[0].tolist()
    return hits, best_root_col


def load_baseband(filename: str) -> np.ndarray:
    """
    Read the bin file, format Q15 - 15 bits fixed-point numbers interleaved
    real and image, one bit for sign. Maximal number for 15 bits is 32768.

    The task description says the signal is oversampled by 4 and has to be
    downsampled, but the sample count (16 + 128) * 9 = 1296 and the spectrum
    (fully occupied, not a quarter) say otherwise, so decimate only when the
    energy really is confined to the central quarter.
    """
    iq = np.fromfile(filename, dtype=np.int16).astype(np.float64) / 32768.0
    rx = iq[0::2] + 1j * iq[1::2]

    P = np.abs(np.fft.fftshift(np.fft.fft(rx * np.hanning(len(rx))))) ** 2
    if P[3 * len(P) // 8 : 5 * len(P) // 8].sum() / P.sum() > 0.8:
        cutoff = 0.9 * (FS_CAPTURE / DECIM) / 2.0  # 0.9 * 960 kHz
        taps = signal.firwin(numtaps=65, cutoff=cutoff, fs=FS_CAPTURE)
        rx = signal.lfilter(taps, 1.0, rx)[::DECIM]
    return rx


def plot_detection(corrs, scores, detected_root, pilot_syms, frame_start):
    """Correlation traces (top) and per-symbol root scores (bottom)."""
    fig, (ax_c, ax_s) = plt.subplots(2, 1, figsize=(11, 7))

    for root, mag in corrs.items():
        hit = root == detected_root
        ax_c.plot(mag, lw=1.2 if hit else 0.7, alpha=1.0 if hit else 0.45,
                  label=f"root {root}{' (detected)' if hit else ''}")
    mag = corrs[detected_root]
    peaks = [frame_start + s * SYM_LEN + CP_LEN for s in pilot_syms]
    peaks = [p for p in peaks if 0 <= p < len(mag)]
    if peaks:
        ax_c.plot(peaks, mag[peaks], "rv", ms=9,
                  label=f"pilot symbols {pilot_syms}")
    ax_c.set(xlabel=f"sample index @ {FS_NATIVE / 1e6:g} MHz",
             ylabel="normalized |xcorr|", ylim=(0, 1.05),
             title="Cross-correlation with candidate ZC pilot symbols")

    for i, root in enumerate(PILOT_ROOTS):
        ax_s.plot(scores[:, i], "o-", lw=0.9, label=f"root {root}")
    ax_s.axhline(0.8, color="grey", ls="--", lw=0.8, label="threshold")
    ax_s.set(xlabel="OFDM symbol index", ylabel="subcarrier match",
             ylim=(0, 1.05), title=f"Per-symbol ZC score (frame start "
                                   f"{frame_start})")

    for ax in (ax_c, ax_s):
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("zc_correlation.png", dpi=130)


rx_bb = load_baseband(path)

#   let's form Zadoff-Chu sequences and calculate correaltion for all symbols and all roots
corrs, best_root, best_start = search_zc(rx_bb)

# The peak sits at the start of the useful part, so back off the CP to get the
# frame start; the frame is not necessarily aligned at sample 0.
frame_start = (best_start - CP_LEN) % SYM_LEN
scores = per_symbol_zc_scores(rx_bb, timing=frame_start)
pilot_syms, root_col = find_pilot_symbol_indices(scores)
detected_root = PILOT_ROOTS[root_col]

#   print results to console
print(f"matched-filter root {best_root}, peak at sample {best_start}")
print(f"frame start {frame_start}, root {detected_root}, "
      f"pilot symbols {pilot_syms}")

#   plot correlation
plot_detection(corrs, scores, detected_root, pilot_syms, frame_start)












