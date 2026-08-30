# -*- coding: utf-8 -*-
"""
Передатчик (MacBook + AntSDR E310).

Кадр загружается в железо один раз и крутится циклически (tx_cyclic_buffer):
хост после запуска не участвует, приёмник ловит буфер с произвольной точки.

Примеры:
    python tx_e310.py --uri ip:192.168.1.10 --freq 2.4e9 --atten 10
    python tx_e310.py --driver ad9361 --uri ip:192.168.1.10
    python tx_e310.py --save frame.npy            # только сохранить кадр, без железа

Прошивка Pluto -> --driver pluto (по умолчанию), родная прошивка E310 -> --driver ad9361.
"""

import argparse
import numpy as np
from zc_ofdm_common import Cfg, build_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ip:192.168.1.10")
    ap.add_argument("--driver", choices=["pluto", "ad9361"], default="pluto")
    ap.add_argument("--freq", type=float, default=2.4e9, help="несущая, Гц")
    ap.add_argument("--atten", type=float, default=10.0,
                    help="ослабление TX, дБ (0 = максимальная мощность)")
    ap.add_argument("--save", default=None, help="сохранить кадр в .npy и выйти")
    args = ap.parse_args()

    cfg = Cfg()
    ref = build_frame(cfg)
    frame = ref.frame

    # масштаб: пик ~ четверть полной шкалы int16 (2**14) — запас под PAPR и DAC
    scale = 2 ** 14 * 0.25 / np.max(np.abs(frame))
    iq = (frame * scale).astype(np.complex64)

    print(f"кадр: {len(frame)} отсчётов = {len(frame) / cfg.FS * 1e3:.3f} мс, "
          f"fs = {cfg.FS / 1e6:.2f} МГц, PAPR = "
          f"{10 * np.log10(np.max(np.abs(frame))**2 / np.mean(np.abs(frame)**2)):.1f} дБ")

    if args.save:
        np.save(args.save, iq)
        print(f"сохранено в {args.save}")
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

    print(f"передача запущена на {args.freq / 1e6:.1f} МГц, ослабление {args.atten} дБ. "
          f"Ctrl+C для остановки.")
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        sdr.tx_destroy_buffer()
        print("остановлено")


if __name__ == "__main__":
    main()
