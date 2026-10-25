"""Plot SRK gNB/UE statistics collected from docker logs (gnb_stats.log, ue_csi.log, ping.log).

Usage: python plot_stats.py <stats_dir>   -> <stats_dir>/stats_2x2.png, summary.csv
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(sys.argv[1])
BLUE, ORANGE = "#2a78d6", "#eb6834"   # categorical slots 1, 2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ts(s):
    return datetime.fromisoformat(s[:26] + "+00:00").timestamp()  # docker --timestamps are UTC


g = {k: [] for k in ("dl", "ul", "mac", "csi", "rsrp")}
for line in open(D / "gnb_stats.log"):
    t = ts(line)
    if m := re.search(r"dlsch_rounds (\d+)/(\d+)/(\d+)/(\d+), dlsch_errors (\d+).*BLER ([\d.]+) MCS \(\d\) (\d+)", line):
        g["dl"].append((t, *map(float, m.groups())))
    elif m := re.search(r"ulsch_rounds (\d+)/(\d+)/(\d+)/(\d+), ulsch_errors (\d+).*BLER ([\d.]+) MCS \(\d\) (\d+).*SNR ([-\d.]+) dB", line):
        g["ul"].append((t, *map(float, m.groups())))
    elif m := re.search(r"MAC:\s+TX\s+(\d+) RX\s+(\d+) bytes", line):
        g["mac"].append((t, float(m[1]), float(m[2])))
    elif m := re.search(r"CQI (\d+), RI (\d+)", line):
        g["csi"].append((t, float(m[1]), float(m[2])))
    elif m := re.search(r"PH (-?\d+) dB.*average RSRP (-?\d+)", line):
        g["rsrp"].append((t, float(m[1]), float(m[2])))
g = {k: np.array(v) for k, v in g.items()}

ue = []
for line in open(D / "ue_csi.log"):
    if m := re.search(r"RSRP = (-?\d+) dBm, RI = (\d+).*SINR = (-?\d+) dB, CQI = (\d+)", line):
        ue.append((ts(line), *map(float, m.groups())))
ue = np.array(ue)

ping = []
for line in open(D / "ping.log"):
    if m := re.search(r"\[([\d.]+)\].*icmp_seq=(\d+).*time=([\d.]+) ms", line):
        ping.append(tuple(map(float, m.groups())))
ping = np.array(ping)

t0 = g["dl"][0, 0]
rel = lambda a: (a - t0) / 60.0  # minutes


def smooth(y, n=8):  # rolling mean over n samples, for readability of noisy per-interval series
    k = np.ones(n) / n
    return np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), k, mode="same")


def rate(a, col):  # kbit/s from cumulative byte counters
    dt = np.diff(a[:, 0])
    return a[1:, 0], np.diff(a[:, col]) * 8 / dt / 1e3


def interval_bler(a):  # BLER per reporting interval from cumulative HARQ rounds
    tx1 = np.diff(a[:, 1])
    retx = np.diff(a[:, 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        return a[1:, 0], np.where(tx1 > 0, retx / tx1, np.nan) * 100


plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4})
fig, ax = plt.subplots(3, 2, figsize=(12, 9), sharex=True, constrained_layout=True)
dur = (g["dl"][-1, 0] - t0) / 60
fig.suptitle(f"SRK 5G NR 2x2 MIMO, gNB U220 + UE E316, 3604.8 MHz, 24 PRB — {dur:.1f} мин", color=INK)

a = ax[0, 0]
t, dl = rate(g["mac"], 1); _, ul = rate(g["mac"], 2)
a.plot(rel(t), smooth(dl), color=BLUE, label="DL (MAC TX gNB)")
a.plot(rel(t), smooth(ul), color=ORANGE, label="UL (MAC RX gNB)")
a.set_title("Трафик, кбит/с (скольз. среднее ~10 с)"); a.legend(frameon=False)

a = ax[0, 1]
t, b = interval_bler(g["dl"]); a.plot(rel(t), smooth(b), color=BLUE, label="DL")
t, b = interval_bler(g["ul"]); a.plot(rel(t), smooth(b), color=ORANGE, label="UL")
a.set_title("BLER первой передачи, % (скольз. среднее ~10 с)"); a.legend(frameon=False)

a = ax[1, 0]
a.plot(rel(ue[:, 0]), smooth(ue[:, 3], 100), color=BLUE, label="DL SINR, CSI-RS (UE)")
a.plot(rel(g["ul"][:, 0]), smooth(g["ul"][:, 8]), color=ORANGE, label="UL SNR, PUSCH (gNB)")
a.set_title("SINR / SNR, дБ (скольз. среднее ~10 с)"); a.legend(frameon=False)

a = ax[1, 1]
a.plot(rel(ue[:, 0]), ue[:, 1], color=BLUE)
a.set_title("SSB-RSRP по отчёту UE, dBm (без калибровки E316)")

a = ax[2, 0]
a.step(rel(g["csi"][:, 0]), g["csi"][:, 1], where="post", color=BLUE, label="CQI")
a.step(rel(g["csi"][:, 0]), g["csi"][:, 2], where="post", color=ORANGE, label="RI")
a.set_title("CQI и RI (CSI-отчёты в gNB)"); a.legend(frameon=False); a.set_xlabel("минуты")

a = ax[2, 1]
a.plot(rel(ping[:, 0]), ping[:, 2], color=BLUE, linewidth=0.8)
a.set_title("RTT ping UE → ext-dn, мс"); a.set_xlabel("минуты")

out = D / "stats_2x2.png"
fig.savefig(out, dpi=130, facecolor="#fcfcfb")

seq = ping[:, 1]
lost = int(seq.max() - seq.min() + 1 - len(seq))
dl_last, ul_last = g["dl"][-1], g["ul"][-1]
rows = [
    ("duration_min", f"{dur:.1f}"),
    ("dl_rounds_1/2/3/4", "/".join(f"{x:.0f}" for x in dl_last[1:5])), ("dl_errors", f"{dl_last[5]:.0f}"),
    ("ul_rounds_1/2/3/4", "/".join(f"{x:.0f}" for x in ul_last[1:5])), ("ul_errors", f"{ul_last[5]:.0f}"),
    ("dl_mcs_median", f"{np.median(g['dl'][:, 7]):.0f}"), ("ul_mcs_median", f"{np.median(g['ul'][:, 7]):.0f}"),
    ("ue_sinr_db_median", f"{np.median(ue[:, 3]):.0f}"), ("ue_sinr_db_p5_p95", f"{np.percentile(ue[:, 3], 5):.0f}..{np.percentile(ue[:, 3], 95):.0f}"),
    ("ul_snr_db_median", f"{np.median(g['ul'][:, 8]):.1f}"), ("ul_snr_db_p5_p95", f"{np.percentile(g['ul'][:, 8], 5):.1f}..{np.percentile(g['ul'][:, 8], 95):.1f}"),
    ("rsrp_dbm_median", f"{np.median(ue[:, 1]):.0f}"),
    ("ri2_share_pct", f"{100 * np.mean(ue[:, 2] == 2):.1f}"), ("cqi_median", f"{np.median(ue[:, 4]):.0f}"),
    ("ping_sent", f"{len(seq) + lost}"), ("ping_lost", f"{lost}"),
    ("rtt_ms_median_p95", f"{np.median(ping[:, 2]):.1f}/{np.percentile(ping[:, 2], 95):.1f}"),
    ("dl_kbps_mean", f"{dl.mean():.1f}"), ("ul_kbps_mean", f"{ul.mean():.1f}"),
]
with open(D / "summary.csv", "w") as f:
    for k, v in rows:
        f.write(f"{k},{v}\n")
print(out)
for k, v in rows:
    print(f"{k:22s} {v}")
