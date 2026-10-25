"""Capture from ANTSDR E316 and search NR SSB (PSS/SSS) of the SRK gNB.

Usage: python ssb_capture.py [freq_hz] [out.npy]
       python ssb_capture.py capture.cf32   (analyse a complex64 file at 15.36 MSps)
"""
import sys
import numpy as np

IN_FILE = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".cf32") else None
FC = float(sys.argv[1]) if len(sys.argv) > 1 and not IN_FILE else 3604.8e6
OUT = sys.argv[2] if len(sys.argv) > 2 else None
FS = 15.36e6
NFFT = 512          # 30 kHz SCS at 15.36 MSps
CP = 36
N = int(0.1 * FS)   # 100 ms, covers 5 SSB periods of 20 ms


def mseq(taps, init):
    x = list(init)
    for i in range(127 - 7):
        x.append((x[i + taps[0]] + x[i]) % 2)
    return np.array(x)


X_PSS = mseq((4,), [0, 1, 1, 0, 1, 1, 1])
X0 = mseq((4,), [1, 0, 0, 0, 0, 0, 0])
X1 = mseq((1,), [1, 0, 0, 0, 0, 0, 0])


def pss(nid2):
    n = np.arange(127)
    return 1 - 2 * X_PSS[(n + 43 * nid2) % 127]


def sss(nid1, nid2):
    n = np.arange(127)
    m0 = 15 * (nid1 // 112) + 5 * nid2
    m1 = nid1 % 112
    return (1 - 2 * X0[(n + m0) % 127]) * (1 - 2 * X1[(n + m1) % 127])


def to_time(seq):
    # PSS/SSS occupy SSB subcarriers 56..182; SSB centre is subcarrier 120 -> k = -64..62
    f = np.zeros(NFFT, complex)
    k = np.arange(127) - 64
    f[k % NFFT] = seq
    return np.fft.ifft(f) * np.sqrt(NFFT)


def xcorr(y, ref):
    n = len(y) + len(ref) - 1
    L = 1 << (n - 1).bit_length()
    c = np.fft.ifft(np.fft.fft(y, L) * np.conj(np.fft.fft(ref, L)))
    return c[: len(y) - len(ref) + 1]


def norm_corr(y, ref):
    c = np.abs(xcorr(y, ref)) / np.linalg.norm(ref)
    e = np.sqrt(np.convolve(np.abs(y) ** 2, np.ones(len(ref)), "valid")) + 1e-9
    return c / e


def capture():
    if IN_FILE:
        return np.fromfile(IN_FILE, np.complex64)[: N] * 2048
    import adi
    sdr = adi.ad9361("ip:192.168.2.2")
    sdr.rx_enabled_channels = [0]
    sdr.sample_rate = int(FS)
    sdr.rx_rf_bandwidth = int(10e6)
    sdr.rx_lo = int(FC)
    sdr.gain_control_mode_chan0 = "slow_attack"
    sdr.rx_buffer_size = N
    for _ in range(3):
        sdr.rx()  # flush stale buffers after retune
    x = sdr.rx().astype(np.complex64)
    print(f"LO {sdr.rx_lo/1e6:.3f} MHz, gain {sdr.rx_hardwaregain_chan0} dB, "
          f"rms {np.sqrt(np.mean(np.abs(x)**2)):.1f} (full scale 2048)")
    return x


def main():
    x = capture()
    if OUT:
        np.save(OUT, x)
    x = x - x.mean()
    t = np.arange(len(x)) / FS
    best = None
    W = int(0.021 * FS)  # one 20 ms SSB period is enough for the search
    for cfo in np.arange(-60e3, 60.1e3, 2.5e3):
        y = x[:W] * np.exp(-2j * np.pi * cfo * t[:W])
        for nid2 in range(3):
            m = norm_corr(y, to_time(pss(nid2)))
            i = int(np.argmax(m))
            if best is None or m[i] > best[0]:
                best = (m[i], cfo, nid2)
    _, cfo, nid2 = best
    m = norm_corr(x * np.exp(-2j * np.pi * cfo * t), to_time(pss(nid2)))
    peak = m.max()
    print(f"PSS: N_ID2={nid2}, CFO {cfo/1e3:+.1f} kHz, normalized peak {peak:.2f} "
          f"(noise median {np.median(m):.3f})")
    # all PSS occurrences (expect ~20 ms period)
    hits = np.where(m > 0.5 * peak)[0]
    groups = []
    for h in hits:
        if not groups or h - groups[-1][-1] > NFFT:
            groups.append([h])
        else:
            groups[-1].append(h)
    starts = [g[int(np.argmax(m[g]))] for g in groups]
    print("PSS positions (ms):", [round(s / FS * 1e3, 3) for s in starts])
    if len(starts) > 1:
        print("period (ms):", np.round(np.diff(starts) / FS * 1e3, 3))

    # SSS: symbol 2 after PSS (symbol 0), try nominal CP and long-CP shift
    y = x * np.exp(-2j * np.pi * cfo * t)
    k = (np.arange(127) - 64) % NFFT
    scores = np.zeros(336)
    for s in starts:
        p_f = np.fft.fft(y[s:s + NFFT]) / np.sqrt(NFFT)
        h = p_f[k] / pss(nid2)  # channel estimate from PSS
        for off in (2 * (NFFT + CP), 2 * (NFFT + CP) + 8, 2 * (NFFT + CP) - 8):
            seg = y[s + off:s + off + NFFT]
            if len(seg) < NFFT:
                continue
            s_f = np.fft.fft(seg) / np.sqrt(NFFT)
            z = s_f[k] * np.conj(h)
            for nid1 in range(336):
                v = abs(np.vdot(sss(nid1, nid2), z))
                scores[nid1] = max(scores[nid1], v)
    nid1 = int(np.argmax(scores))
    srt = np.sort(scores)
    print(f"SSS: N_ID1={nid1}, score {srt[-1]:.1f} vs next best {srt[-2]:.1f}")
    print(f"=> PCI = 3*{nid1}+{nid2} = {3 * nid1 + nid2}")


if __name__ == "__main__":
    main()
