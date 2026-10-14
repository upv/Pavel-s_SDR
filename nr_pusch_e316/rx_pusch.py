#!/usr/bin/env python3
"""Receive the WirelessNN PUSCH slot on the ANTSDR E316.

    python -m nr_pusch_e316.rx_pusch --freq 3.5e9 --gain 40               # live capture
    python -m nr_pusch_e316.rx_pusch --file cap.npy                        # a saved capture
    python -m nr_pusch_e316.rx_pusch --sim --snr 15 --cfo 4000             # synthetic check

Run with the WirelessNN virtual environment (Sionna / PyTorch).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .receiver import process_capture, simulate_capture, summarize
from .waveform import from_int16_iq, make_frame
from .wirelessnn import LinkConfig, build_link


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=None, help="libiio URI (default: first USB Pluto/ANTSDR, else ip:192.168.2.2)")
    ap.add_argument("--freq", type=float, default=3.5e9, help="carrier [Hz]")
    ap.add_argument("--gain", default="40", help="RX gain dB, 'agc' or 'auto' (AGC once, then frozen)")
    ap.add_argument("--bw", type=float, default=28e6, help="RX analog bandwidth [Hz]")
    ap.add_argument("--rx-ant", type=int, default=1, choices=(1, 2), help="RX1 only or RX1+RX2")
    ap.add_argument("--nsamp", type=int, default=1 << 17, help="samples per capture (7 frames at 2^17)")
    ap.add_argument("--captures", type=int, default=1)
    ap.add_argument("--mod", default="qpsk", choices=("qpsk", "qam16", "qam64"))
    ap.add_argument("--prb", type=int, default=68)
    ap.add_argument("--seed", type=int, default=42, help="seed of the transmitted transport block")
    ap.add_argument("--estimators", default="ls_lin,ls_soft_window")
    ap.add_argument("--receivers", default="lmmse")
    ap.add_argument("--ta", type=int, default=16, help="FFT window advance into the CP [samples]")
    ap.add_argument("--file", default=None, help="offline: .npy capture (int16 interleaved or complex [rx, n])")
    ap.add_argument("--save-raw", default=None, help="save the raw int16 capture(s) to this .npy")
    ap.add_argument("--sim", action="store_true", help="synthetic capture instead of hardware")
    ap.add_argument("--snr", type=float, default=20.0, help="--sim: SNR per sample [dB]")
    ap.add_argument("--cfo", type=float, default=3000.0, help="--sim: carrier offset [Hz]")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefix", default="rx_pusch", help="prefix of the plot / json files")
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args(argv)


def load_capture(path: str, num_rx: int) -> np.ndarray:
    a = np.load(path)
    if a.dtype == np.int16:
        return from_int16_iq(a, num_rx)
    a = np.asarray(a, dtype=np.complex64)
    return np.atleast_2d(a)


def main(argv=None) -> int:
    args = parse_args(argv)
    link = LinkConfig(modulation=args.mod, num_prb=args.prb, num_rx_ant=args.rx_ant, seed=args.seed,
                      estimators=tuple(args.estimators.split(",")), receivers=tuple(args.receivers.split(",")))
    cfg, tx, rx = build_link(link)
    frame = make_frame(link, tx, rx, cfg)
    verbose = not args.quiet
    if verbose:
        print(f"PUSCH {args.mod} {args.prb} PRB, fs={frame.fs/1e6:.2f} MSps, slot {frame.slot_len} + gap {frame.gap} samples, "
              f"TB {frame.bits.size} bits, DMRS symbols {frame.dmrs_symbols}")

    captures = []
    if args.sim:
        captures.append(simulate_capture(frame, snr_db=args.snr, cfo_hz=args.cfo, num_rx=args.rx_ant))
    elif args.file:
        captures.append(load_capture(args.file, args.rx_ant))
    else:
        from .sdr import E316

        sdr = E316(args.uri)
        gain = args.gain if args.gain in ("agc", "auto") else float(args.gain)
        g = sdr.configure_rx(args.freq, frame.fs, args.bw, gain, channels=tuple(range(args.rx_ant)))
        if verbose:
            print(f"{sdr.description}: RX {args.freq/1e6:.1f} MHz gain {g:.0f} dB bw {args.bw/1e6:.0f} MHz")
        raws = []
        for i in range(args.captures):
            raw = sdr.capture(args.nsamp, flush=4)
            raws.append(raw)
            cap = from_int16_iq(raw, args.rx_ant)
            if verbose:
                print(f"capture {i}: {cap.shape[1]} samples, rms={np.sqrt(np.mean(np.abs(cap)**2)):.4f} "
                      f"peak={np.abs(cap).max():.3f} of 1.0 FS")
            captures.append(cap)
        sdr.close()
        if args.save_raw:
            np.save(args.save_raw, np.stack(raws) if len(raws) > 1 else raws[0])

    results = []
    for cap in captures:
        results += process_capture(cap, frame, rx, cfg, tx=tx, timing_advance=args.ta, verbose=verbose)
    summary = summarize(results)
    print("\nsummary over", len(results), "slots:")
    for (est, rec), s in summary.items():
        print(f"  {est:16s} {rec:8s} BER={s['ber']:.2e} ({s['bit_errors']} bits)  BLER={s['bler']:.2f} ({s['block_errors']}/{s['slots']})")
    with open(f"{args.prefix}_summary.json", "w") as f:
        json.dump({f"{e}/{r}": v for (e, r), v in summary.items()}, f, indent=1)
    if args.plot and results:
        plot_results(captures[0], results, frame, args.prefix)
    return 0 if results else 1


def plot_results(cap: np.ndarray, results, frame, prefix: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.atleast_2d(cap)[0]
    res = results[0]
    est = next(iter(res.h_hat))
    h = res.h_hat[est][0]                                       # [sym, used sc]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    m = res.sync.extra["metric"]
    ax[0, 0].plot(m, lw=0.5)
    for q in results:
        ax[0, 0].axvline(q.sync.extra["peak"], color="r", lw=0.7, ls="--")
    ax[0, 0].set(title="DMRS correlation metric", xlabel="sample", ylim=(0, 1.05))
    f = np.fft.fftshift(np.fft.fftfreq(4096, 1 / frame.fs)) / 1e6
    seg = r[:len(r) // 4096 * 4096].reshape(-1, 4096) * np.hanning(4096)
    P = 10 * np.log10(np.mean(np.abs(np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)) ** 2, axis=0) + 1e-20)
    ax[0, 1].plot(f, P, lw=0.6); ax[0, 1].set(title="capture spectrum", xlabel="MHz", ylabel="dB")
    ax[0, 2].plot(np.abs(r[res.sync.slot_start - 200: res.sync.slot_start + frame.frame_len + 200]), lw=0.4)
    ax[0, 2].axvline(200, color="r", lw=0.7); ax[0, 2].axvline(200 + frame.slot_len, color="r", lw=0.7)
    ax[0, 2].set(title=f"|r| around slot 0 (CFO {res.sync.cfo_hz:+.0f} Hz, SNR~{res.snr_db:.1f} dB)", xlabel="sample")
    sc = (np.arange(h.shape[1]) - h.shape[1] / 2) * frame.fs / frame.fft_size / 1e6
    ax[1, 0].plot(sc, 20 * np.log10(np.abs(h[frame.dmrs_symbols[0]]) + 1e-12), lw=0.7, label=f"sym {frame.dmrs_symbols[0]}")
    ax[1, 0].plot(sc, 20 * np.log10(np.abs(h[frame.dmrs_symbols[-1]]) + 1e-12), lw=0.7, label=f"sym {frame.dmrs_symbols[-1]}")
    ax[1, 0].set(title=f"|H(f)| dB ({est})", xlabel="MHz from carrier"); ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(sc, np.unwrap(np.angle(h[frame.dmrs_symbols[0]])), lw=0.7)
    ax[1, 1].set(title="phase H(f) [rad]", xlabel="MHz from carrier")
    if res.eq_symbols:
        e = res.eq_symbols[est]
        ax[1, 2].plot(e.real, e.imag, ".", ms=1.5, alpha=0.3)
        ax[1, 2].set(title=f"equalised data REs, EVM {res.evm_db[est]:.1f} dB", xlim=(-2, 2), ylim=(-2, 2), aspect="equal")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(f"{frame.link.modulation.upper()} {frame.link.num_prb} PRB  slot start {res.sync.slot_start}  "
                 f"bit errors {res.bit_errors}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{prefix}_result.png", dpi=120)
    print(f"plot: {prefix}_result.png")


if __name__ == "__main__":
    sys.exit(main())
