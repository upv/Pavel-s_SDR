#!/usr/bin/env python3
"""Transmit the WirelessNN PUSCH frame cyclically on the ANTSDR E316 TX1.

    python -m nr_pusch_e316.tx_pusch --freq 3.5e9 --atten 30 --seconds 60
    python -m nr_pusch_e316.tx_pusch --save frame_qpsk.npz --no-tx       # only write the frame

Same --mod/--prb/--seed as the receiver -> same transport block.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .waveform import make_frame, to_int16_iq
from .wirelessnn import LinkConfig, build_link


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default=None)
    ap.add_argument("--freq", type=float, default=3.5e9)
    ap.add_argument("--atten", type=float, default=30.0, help="TX attenuation [dB], 0 = full power")
    ap.add_argument("--bw", type=float, default=28e6)
    ap.add_argument("--mod", default="qpsk", choices=("qpsk", "qam16", "qam64"))
    ap.add_argument("--prb", type=int, default=68)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--peak", type=float, default=0.9, help="DAC peak as a fraction of full scale")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")
    ap.add_argument("--save", default=None, help="write the frame (.npz) for offline use")
    ap.add_argument("--tx", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args(argv)

    link = LinkConfig(modulation=args.mod, num_prb=args.prb, seed=args.seed)
    cfg, tx, rx = build_link(link)
    frame = make_frame(link, tx, rx, cfg)
    w = frame.waveform
    papr = 10 * np.log10(np.abs(w).max() ** 2 / np.mean(np.abs(frame.slot) ** 2))
    print(f"frame: slot {frame.slot_len} + gap {frame.gap} = {frame.frame_len} samples "
          f"({frame.frame_len / frame.fs * 1e6:.0f} us) @ {frame.fs/1e6:.2f} MSps, PAPR {papr:.1f} dB, "
          f"TB {frame.bits.size} bits")
    if args.save:
        frame.save(args.save)
        print(f"saved {args.save}")
    if not args.tx:
        return 0

    from .sdr import E316

    sdr = E316(args.uri)
    sdr.configure_tx(args.freq, frame.fs, args.bw, args.atten)
    sdr.transmit_cyclic(to_int16_iq(w, args.peak))
    print(f"{sdr.description}: TX {args.freq/1e6:.1f} MHz, attenuation {args.atten:.1f} dB, cyclic ... Ctrl-C to stop")
    t0 = time.time()
    try:
        while args.seconds <= 0 or time.time() - t0 < args.seconds:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    sdr.stop_tx()
    sdr.close()
    print("TX stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
