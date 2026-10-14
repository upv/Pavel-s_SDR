#!/usr/bin/env python3
"""Self-test without antennas: AD9361 digital loopback (TX data -> RX data inside the chip).

Nothing is radiated: the ``loopback`` debug attribute routes the DAC samples back
into the RX data path and the TX attenuation is set to its maximum. The test checks
the whole on-device path (cyclic DMA buffer, 2R2T channel mapping, sample rate,
int16 formats) and the receiver chain on real captures.

    python -m nr_pusch_e316.selftest [--uri ...] [--mod qpsk] [--prb 68]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .receiver import process_capture, summarize
from .sdr import E316
from .waveform import from_int16_iq, make_frame, to_int16_iq
from .wirelessnn import LinkConfig, build_link


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=None)
    ap.add_argument("--mod", default="qpsk", choices=("qpsk", "qam16", "qam64"))
    ap.add_argument("--prb", type=int, default=68)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nsamp", type=int, default=1 << 17)
    ap.add_argument("--peak", type=float, default=0.5)
    ap.add_argument("--save-raw", default=None)
    args = ap.parse_args(argv)

    link = LinkConfig(modulation=args.mod, num_prb=args.prb, seed=args.seed)
    cfg, tx, rx = build_link(link)
    frame = make_frame(link, tx, rx, cfg)
    sdr = E316(args.uri)
    print(sdr.description)
    fc = 2.0e9
    try:
        sdr.configure_tx(fc, frame.fs, 28e6, attenuation_db=89.75)      # max attenuation, no RF
        sdr.configure_rx(fc, frame.fs, 28e6, gain=0.0)
        sdr.phy.debug_attrs["loopback"].value = "1"                      # digital TX -> RX
        sdr.transmit_cyclic(to_int16_iq(frame.waveform, args.peak))
        time.sleep(0.2)
        raw = sdr.capture(args.nsamp, flush=6)
    finally:
        sdr.stop_tx()
        try:
            sdr.phy.debug_attrs["loopback"].value = "0"
        finally:
            sdr.close()
    if args.save_raw:
        np.save(args.save_raw, raw)
    cap = from_int16_iq(raw, 1)
    print(f"capture: {cap.shape[1]} samples rms={np.sqrt(np.mean(np.abs(cap)**2)):.4f} peak={np.abs(cap).max():.3f}")
    res = process_capture(cap, frame, rx, cfg, tx=tx)
    s = summarize(res)
    ok = bool(res) and all(v["bit_errors"] == 0 for v in s.values())
    print("summary:", {f"{e}/{r}": (v["bit_errors"], v["bler"]) for (e, r), v in s.items()})
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
