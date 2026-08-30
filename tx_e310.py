# -*- coding: utf-8 -*-
"""
Transmitter (MacBook + AntSDR E310).

The frame is loaded once and loops in hardware (tx_cyclic_buffer):
the host is idle after start; the receiver grabs a buffer from an arbitrary point.

Examples:
    python tx_e310.py --uri ip:192.168.1.10 --freq 2.4e9 --atten 10
    python tx_e310.py --driver ad9361 --uri ip:192.168.1.10
    python tx_e310.py --save frame.npy            # save frame only, no hardware

Pluto firmware -> --driver pluto (default), native E310 firmware -> --driver ad9361.
"""

import argparse
import numpy as np
from zc_ofdm_common import Cfg, build_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ip:192.168.1.10")
    ap.add_argument("--driver", choices=["pluto", "ad9361"], default="pluto")
    ap.add_argument("--freq", type=float, default=2.4e9, help="carrier, Hz")
    ap.add_argument("--atten", type=float, default=10.0,
                    help="TX attenuation, dB (0 = max power)")
    ap.add_argument("--save", default=None, help="save frame to .npy and exit")
    args = ap.parse_args()

    cfg = Cfg()
    ref = build_frame(cfg)
    frame = ref.frame

    # scale: peak ~ quarter of int16 full scale (2**14) — headroom for PAPR and DAC
    scale = 2 ** 14 * 0.25 / np.max(np.abs(frame))
    iq = (frame * scale).astype(np.complex64)

    print(f"frame: {len(frame)} samples = {len(frame) / cfg.FS * 1e3:.3f} ms, "
          f"fs = {cfg.FS / 1e6:.2f} MHz, PAPR = "
          f"{10 * np.log10(np.max(np.abs(frame))**2 / np.mean(np.abs(frame)**2)):.1f} dB")

    if args.save:
        np.save(args.save, iq)
        print(f"saved to {args.save}")
        return

    import adi
    if args.driver == "pluto":
        sdr = adi.Pluto(uri=args.uri)
    else:
        sdr = adi.ad9361(uri=args.uri)

    sdr.sample_rate = int(cfg.FS)
    sdr.tx_rf_bandwidth = int(cfg.FS)
    sdr.tx_lo = int(args.freq)
    if args.driver == "pluto":
        sdr.tx_hardwaregain_chan0 = -abs(args.atten)
        sdr.tx_cyclic_buffer = True
        sdr.tx(iq)
    else:
        sdr.tx_enabled_channels = [0]
        sdr.tx_hardwaregain_chan0 = -abs(args.atten)
        sdr.tx_cyclic_buffer = True
        sdr.tx(iq)

    print(f"TX started at {args.freq / 1e6:.1f} MHz, attenuation {args.atten} dB. "
          f"Ctrl+C to stop.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()
        print("stopped")


if __name__ == "__main__":
    main()
