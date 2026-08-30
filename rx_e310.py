# -*- coding: utf-8 -*-
"""
Приёмник (Orange Pi AIpro + второй AntSDR E310).

Захватывает буфер IQ (или читает записанный .npy), прогоняет цепочку:
Шмидл-Кокс -> дробный CFO -> целочисленный CFO -> точный тайминг по ZC ->
LS-оценка канала -> фазовый трекинг -> задержка -> эквализация -> EVM.

Примеры:
    python rx_e310.py --uri usb: --freq 2.4e9 --gain 40 --save-raw cap.npy --plot
    python rx_e310.py --uri ip:192.168.2.1 --freq 2.4e9 --gain 40 --save-raw cap.npy --plot
    python rx_e310.py --file cap.npy --plot          # оффлайн-обработка записи
"""

import argparse
import sys
import numpy as np
from zc_ofdm_common import Cfg, build_frame, process_capture, summary

# pyadi-iio: conda env often lacks the pip package; local clone is what the rest of this repo uses
for _p in ("/home/pavel/pyadi-iio/", "/home/pavel/libiio"):
    if _p not in sys.path:
        sys.path.append(_p)


def capture(args, cfg):
    import adi
    if args.driver == "pluto":
        sdr = adi.Pluto(uri=args.uri)
    else:
        sdr = adi.ad9361(uri=args.uri)
        sdr.rx_enabled_channels = [0]

    sdr.sample_rate = int(cfg.FS)
    sdr.rx_rf_bandwidth = int(cfg.FS)
    sdr.rx_lo = int(args.freq)
    sdr.gain_control_mode_chan0 = "manual"   # AGC выключен: оценки канала сравнимы между захватами
    sdr.rx_hardwaregain_chan0 = args.gain
    sdr.rx_buffer_size = args.nsamp
    _ = sdr.rx()                             # первый буфер после настройки — выбросить
    rx = sdr.rx().astype(np.complex64)
    sdr.rx_destroy_buffer()
    return rx


def make_plots(rx, res, ref, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = ref.cfg

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    ax[0, 0].plot(res.sc_metric)
    ax[0, 0].axvline(res.coarse_idx, color="r", ls="--", lw=1)
    ax[0, 0].set_title("Метрика Шмидла-Кокса")
    ax[0, 0].set_xlabel("отсчёт")

    ax[0, 1].plot(ref.pilot_offsets * cfg.SCS / 1e6,
                  20 * np.log10(np.abs(res.H) + 1e-9))
    ax[0, 1].set_title("|H(f)|, дБ (усреднён по символам)")
    ax[0, 1].set_xlabel("МГц от несущей")

    ax[1, 0].plot(ref.pilot_offsets * cfg.SCS / 1e6,
                  np.unwrap(np.angle(res.H)))
    ax[1, 0].set_title("фаза H(f), рад")
    ax[1, 0].set_xlabel("МГц от несущей")

    ax[1, 1].plot(res.eq_data.real.ravel(), res.eq_data.imag.ravel(),
                  ".", ms=2, alpha=0.4)
    ax[1, 1].set_title(f"QPSK после эквализации, EVM = {res.evm_db:.1f} дБ")
    ax[1, 1].set_xlim(-2, 2); ax[1, 1].set_ylim(-2, 2)
    ax[1, 1].set_aspect("equal")

    fig.tight_layout()
    fig.savefig(f"{prefix}_result.png", dpi=130)
    print(f"график: {prefix}_result.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", default="ip:192.168.2.1")
    ap.add_argument("--driver", choices=["pluto", "ad9361"], default="pluto")
    ap.add_argument("--freq", type=float, default=2.4e9)
    ap.add_argument("--gain", type=float, default=40.0, help="усиление RX, дБ (AGC выключен)")
    ap.add_argument("--nsamp", type=int, default=1 << 16, help="размер захвата")
    ap.add_argument("--file", default=None, help="оффлайн: .npy с IQ вместо захвата")
    ap.add_argument("--save-raw", default=None, help="сохранить сырой захват в .npy")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cfg = Cfg()
    ref = build_frame(cfg)

    if args.file:
        rx = np.load(args.file)
    else:
        rx = capture(args, cfg)
        if args.save_raw:
            np.save(args.save_raw, rx)
            print(f"сырой захват: {args.save_raw}")

    res = process_capture(rx, ref)
    print(summary(res, cfg))

    if args.plot:
        make_plots(rx, res, ref, prefix=args.file or "rx")


if __name__ == "__main__":
    main()
