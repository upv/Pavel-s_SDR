# -*- coding: utf-8 -*-
"""
Общий модуль для линка E310 <-> E310:
  - параметры OFDM (15 кГц SCS, FFT 1024, fs = 15.36 МГц)
  - генерация Zadoff-Chu (циклическое расширение, как low-PAPR последовательности NR, TS 38.211 5.2.2)
  - сборка кадра: SC-преамбула -> ZC-преамбула точного тайминга -> пилотные символы (comb-2 ZC + QPSK-данные)
  - полная цепочка приёмника: коарс-тайминг + дробный CFO (Шмидл-Кокс), целочисленный CFO
    (дифференциальная частотная корреляция), точный тайминг (кросс-корреляция с ZC),
    LS-оценка канала, остаточный CFO/фазовый трекинг, оценка задержки по наклону фазы, EVM.

Только NumPy; железо подключается в tx_e310.py / rx_e310.py.
"""

from dataclasses import dataclass, field
import numpy as np


# ----------------------------------------------------------------------------- параметры

@dataclass
class Cfg:
    N_FFT: int = 1024
    CP: int = 72                 # упрощение: одинаковый CP у всех символов
    SCS: float = 15e3
    N_ACT: int = 600             # 50 RB, DC не используется
    N_PILOT_SYMS: int = 8        # символы с comb-2 пилотами + данными
    GAP: int = 240               # зазор нулей в конце кадра (кратность 16 для DMA)
    ZC_ROOT_SC: int = 7          # корень ZC для SC-преамбулы
    ZC_ROOT_FINE: int = 25       # корень ZC для преамбулы точного тайминга
    ZC_ROOT_PILOT: int = 11      # корень ZC для пилотов
    DATA_SEED: int = 2026        # приёмник знает данные -> можно мерить EVM
    TIMING_ADVANCE: int = 8      # сдвиг окна FFT внутрь CP (защита от поздних лучей)

    @property
    def FS(self) -> float:
        return self.N_FFT * self.SCS          # 15.36 МГц

    @property
    def SYM_LEN(self) -> int:
        return self.N_FFT + self.CP

    @property
    def N_SYMS(self) -> int:
        return 2 + self.N_PILOT_SYMS          # SC + fine + пилотные

    @property
    def FRAME_LEN(self) -> int:
        return self.N_SYMS * self.SYM_LEN + self.GAP


# ----------------------------------------------------------------------------- утилиты

def largest_prime_below(n: int) -> int:
    def is_prime(m):
        if m < 2:
            return False
        for p in range(2, int(m ** 0.5) + 1):
            if m % p == 0:
                return False
        return True
    m = n
    while not is_prime(m):
        m -= 1
    return m


def zadoff_chu(root: int, length: int) -> np.ndarray:
    """Базовая ZC нечётной простой длины: x_q(m) = exp(-j*pi*q*m*(m+1)/N_zc)."""
    m = np.arange(length)
    return np.exp(-1j * np.pi * root * m * (m + 1) / length)


def zc_extended(root: int, target_len: int) -> np.ndarray:
    """Циклическое расширение ZC простой длины до target_len — как в NR (38.211 5.2.2.1)."""
    n_zc = largest_prime_below(target_len)
    base = zadoff_chu(root, n_zc)
    return base[np.arange(target_len) % n_zc]


def active_offsets(cfg: Cfg) -> np.ndarray:
    """Смещения занятых поднесущих относительно DC: -300..-1, 1..300 (DC пропущен)."""
    half = cfg.N_ACT // 2
    return np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])


def offsets_to_bins(offsets: np.ndarray, cfg: Cfg) -> np.ndarray:
    return np.mod(offsets, cfg.N_FFT)


def ofdm_symbol(freq_vals: np.ndarray, offsets: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Freq -> time с CP; нормировка к единичной средней мощности."""
    X = np.zeros(cfg.N_FFT, dtype=complex)
    X[offsets_to_bins(offsets, cfg)] = freq_vals
    x = np.fft.ifft(X) * cfg.N_FFT / np.sqrt(len(offsets))
    return np.concatenate([x[-cfg.CP:], x])


# ----------------------------------------------------------------------------- сборка кадра

@dataclass
class FrameRef:
    """Всё, что нужно приёмнику: эталоны преамбул, пилотов и данных."""
    cfg: Cfg
    offsets: np.ndarray            # все занятые смещения
    sc_offsets: np.ndarray         # чётные смещения (SC-преамбула)
    pilot_offsets: np.ndarray      # comb-2: каждое второе занятое смещение
    data_offsets: np.ndarray       # остальные
    sc_freq: np.ndarray
    fine_freq: np.ndarray          # ZC на всех занятых поднесущих
    fine_td: np.ndarray            # time-domain полезная часть (без CP) для кросс-корреляции
    pilot_freq: np.ndarray
    data_syms: np.ndarray          # [N_PILOT_SYMS, len(data_offsets)] QPSK
    frame: np.ndarray              # готовый кадр во времени


def build_frame(cfg: Cfg = Cfg()) -> FrameRef:
    offs = active_offsets(cfg)

    # SC-преамбула: только чётные смещения -> две одинаковые половины во времени
    sc_offs = offs[offs % 2 == 0]
    sc_freq = zc_extended(cfg.ZC_ROOT_SC, len(sc_offs))
    sym_sc = ofdm_symbol(sc_freq, sc_offs, cfg)

    # преамбула точного тайминга: ZC на всех 600 поднесущих
    fine_freq = zc_extended(cfg.ZC_ROOT_FINE, len(offs))
    sym_fine = ofdm_symbol(fine_freq, offs, cfg)
    fine_td = sym_fine[cfg.CP:]

    # пилотные символы: comb-2 ZC + QPSK-данные на второй гребёнке
    pilot_offs = offs[0::2]
    data_offs = offs[1::2]
    pilot_freq = zc_extended(cfg.ZC_ROOT_PILOT, len(pilot_offs))

    rng = np.random.default_rng(cfg.DATA_SEED)
    bits = rng.integers(0, 2, size=(cfg.N_PILOT_SYMS, len(data_offs), 2))
    data_syms = ((1 - 2 * bits[..., 0]) + 1j * (1 - 2 * bits[..., 1])) / np.sqrt(2)

    body = [sym_sc, sym_fine]
    for s in range(cfg.N_PILOT_SYMS):
        vals = np.zeros(len(offs), dtype=complex)
        vals[0::2] = pilot_freq
        vals[1::2] = data_syms[s]
        body.append(ofdm_symbol(vals, offs, cfg))

    frame = np.concatenate(body + [np.zeros(cfg.GAP, dtype=complex)])
    # запас по PAPR: пик на ~0.5 полной шкалы оставляем на стороне TX-скрипта
    return FrameRef(cfg, offs, sc_offs, pilot_offs, data_offs,
                    sc_freq, fine_freq, fine_td, pilot_freq, data_syms, frame)


# ----------------------------------------------------------------------------- приёмник

@dataclass
class RxResult:
    coarse_idx: int = 0
    frame_start: int = 0
    cfo_frac_sc: float = 0.0        # дробный CFO (в поднесущих) по Шмидлу-Коксу
    cfo_int_sc: int = 0             # целочисленный CFO (в поднесущих)
    cfo_hz: float = 0.0             # суммарный CFO
    cfo_resid_hz: float = 0.0       # остаточный CFO по фазовому трекингу пилотов
    delay_frac_samples: float = 0.0 # остаточная задержка по наклону фазы H
    snr_db: float = 0.0             # SNR по пилотам
    evm_db: float = 0.0             # EVM данных после эквализации
    H: np.ndarray = field(default_factory=lambda: np.zeros(0))   # усреднённый канал на пилотных поднесущих
    H_syms: np.ndarray = field(default_factory=lambda: np.zeros(0))  # канал по символам [S, K]
    eq_data: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sc_metric: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _schmidl_cox(rx: np.ndarray, cfg: Cfg):
    """Метрика M(d)=|P|^2/R^2 по половинному повторению; возвращает метрику, коарс-индекс, дробный CFO."""
    L = cfg.N_FFT // 2
    a = rx[:-L] if L else rx
    b = rx[L:]
    p = np.conj(a) * b
    q = np.abs(b) ** 2
    win = np.ones(L)
    P = np.convolve(p, win, mode='valid')
    R = np.convolve(q, win, mode='valid') + 1e-12
    M = np.abs(P) ** 2 / R ** 2

    d_max = int(np.argmax(M))
    # середина плато (>=90% максимума вокруг пика) — устойчивее, чем сам argmax
    thr = 0.9 * M[d_max]
    lo = d_max
    while lo > 0 and M[lo - 1] >= thr:
        lo -= 1
    hi = d_max
    while hi < len(M) - 1 and M[hi + 1] >= thr:
        hi += 1
    d = (lo + hi) // 2
    eps_frac = np.angle(P[d]) / np.pi   # в поднесущих, диапазон +-1
    return M, d, eps_frac


def _apply_cfo(rx: np.ndarray, eps_subcarriers: float, cfg: Cfg) -> np.ndarray:
    n = np.arange(len(rx))
    return rx * np.exp(-2j * np.pi * eps_subcarriers * n / cfg.N_FFT)


def _integer_cfo(rx: np.ndarray, fine_start_coarse: int, ref: FrameRef, max_shift: int = 40) -> int:
    """Целочисленный CFO по преамбуле точного тайминга.

    Дифференциальная корреляция по частоте: Z[k] = Y[k]*conj(Y[k+delta]) не зависит от
    линейного набега фазы из-за ошибки тайминга (он даёт общий множитель), поэтому
    работает при коарс-тайминге с ошибкой порядка CP.
    """
    cfg = ref.cfg
    seg = rx[fine_start_coarse:fine_start_coarse + cfg.N_FFT]
    if len(seg) < cfg.N_FFT:
        return 0
    Y = np.fft.fft(seg)

    bins = offsets_to_bins(ref.offsets, cfg)
    X = np.zeros(cfg.N_FFT, dtype=complex)
    X[bins] = ref.fine_freq

    # дифференциалы на соседних занятых бинах (шаг 1 внутри каждой половины спектра)
    d_ref = X[bins] * np.conj(np.roll(X, -1)[bins])
    best, best_val = 0, -1.0
    for s in range(-max_shift, max_shift + 1):
        Ys = np.roll(Y, -s)
        d_rx = Ys[bins] * np.conj(np.roll(Ys, -1)[bins])
        val = np.abs(np.vdot(d_ref, d_rx))
        if val > best_val:
            best_val, best = val, s
    return best


def _fine_timing(rx: np.ndarray, expect: int, ref: FrameRef, search: int) -> int:
    cfg = ref.cfg
    lo = max(0, expect - search)
    hi = min(len(rx) - cfg.N_FFT, expect + search)
    tpl = ref.fine_td
    best, best_val = lo, -1.0
    for d in range(lo, hi):
        val = np.abs(np.vdot(tpl, rx[d:d + cfg.N_FFT]))
        if val > best_val:
            best_val, best = val, d
    return best


def process_capture(rx: np.ndarray, ref: FrameRef) -> RxResult:
    cfg = ref.cfg
    res = RxResult()
    rx = rx.astype(complex) / (np.sqrt(np.mean(np.abs(rx) ** 2)) + 1e-12)

    # --- 1. коарс-тайминг + дробный CFO (Шмидл-Кокс, нечувствителен к CFO)
    M, d, eps_frac = _schmidl_cox(rx, cfg)
    res.sc_metric, res.coarse_idx, res.cfo_frac_sc = M, d, eps_frac
    rx1 = _apply_cfo(rx, eps_frac, cfg)

    # --- 2. целочисленный CFO (до кросс-корреляции с ZC! иначе delay-Doppler смещение пика)
    fine_coarse = d + cfg.SYM_LEN            # коарс-позиция полезной части преамбулы тайминга
    eps_int = _integer_cfo(rx1, fine_coarse, ref)
    res.cfo_int_sc = eps_int
    rx2 = _apply_cfo(rx1, eps_int, cfg)
    res.cfo_hz = (eps_frac + eps_int) * cfg.SCS

    # --- 3. точный тайминг кросс-корреляцией с ZC (теперь CFO скомпенсирован)
    fine_idx = _fine_timing(rx2, fine_coarse, ref, search=2 * cfg.CP)
    frame_start = fine_idx - cfg.SYM_LEN - cfg.CP
    res.frame_start = frame_start
    if frame_start < 0 or frame_start + cfg.FRAME_LEN > len(rx2):
        raise RuntimeError("кадр не помещается в захват целиком: увеличьте rx_buffer_size "
                           "или проверьте, что передатчик работает")

    # --- 4. демодуляция пилотных символов (окно FFT сдвинуто внутрь CP)
    adv = cfg.TIMING_ADVANCE
    pil_bins = offsets_to_bins(ref.pilot_offsets, cfg)
    dat_bins = offsets_to_bins(ref.data_offsets, cfg)
    n_rot = np.exp(2j * np.pi * adv * np.arange(cfg.N_FFT) / cfg.N_FFT)  # компенсация сдвига окна

    H_syms, Y_data = [], []
    for s in range(cfg.N_PILOT_SYMS):
        st = frame_start + (2 + s) * cfg.SYM_LEN + cfg.CP - adv
        Y = np.fft.fft(rx2[st:st + cfg.N_FFT]) * n_rot * np.sqrt(cfg.N_ACT) / cfg.N_FFT
        H_syms.append(Y[pil_bins] / ref.pilot_freq)
        Y_data.append(Y[dat_bins])
    H_syms = np.array(H_syms)
    Y_data = np.array(Y_data)

    # --- 5. фазовый трекинг: общий набег фазы символ-к-символу -> остаточный CFO
    ph = np.angle(np.sum(H_syms * np.conj(H_syms[0]), axis=1))
    ph = np.unwrap(ph)
    sym_t = cfg.SYM_LEN / cfg.FS
    slope = np.polyfit(np.arange(cfg.N_PILOT_SYMS), ph, 1)[0]
    res.cfo_resid_hz = slope / (2 * np.pi * sym_t)
    corr = np.exp(-1j * ph)[:, None]
    H_syms_c = H_syms * corr
    Y_data_c = Y_data * corr

    # --- 6. остаточная задержка: наклон фазы H по частоте (шаг гребёнки = 2 бина)
    pairs = np.sum(H_syms_c[:, 1:] * np.conj(H_syms_c[:, :-1]))
    ph_per_bin = np.angle(pairs) / 2.0
    res.delay_frac_samples = -ph_per_bin * cfg.N_FFT / (2 * np.pi)

    # --- 7. канал, SNR, эквализация данных, EVM
    H = np.mean(H_syms_c, axis=0)
    res.H, res.H_syms = H, H_syms_c
    noise = H_syms_c - H[None, :]
    res.snr_db = 10 * np.log10(np.mean(np.abs(H) ** 2) /
                               (np.mean(np.abs(noise) ** 2) + 1e-15))

    # линейная интерполяция H на поднесущие данных
    H_data = (np.interp(ref.data_offsets, ref.pilot_offsets, H.real) +
              1j * np.interp(ref.data_offsets, ref.pilot_offsets, H.imag))
    eq = Y_data_c / H_data[None, :]
    res.eq_data = eq
    err = eq - ref.data_syms
    res.evm_db = 10 * np.log10(np.mean(np.abs(err) ** 2) /
                               np.mean(np.abs(ref.data_syms) ** 2))
    return res


def summary(res: RxResult, cfg: Cfg) -> str:
    return "\n".join([
        f"коарс-индекс (SC)        : {res.coarse_idx}",
        f"начало кадра             : {res.frame_start}",
        f"CFO дробный / целый      : {res.cfo_frac_sc:+.3f} / {res.cfo_int_sc:+d} поднесущих",
        f"CFO суммарный            : {res.cfo_hz / 1e3:+.3f} кГц",
        f"CFO остаточный (трекинг) : {res.cfo_resid_hz:+.1f} Гц",
        f"остаточная задержка      : {res.delay_frac_samples:+.2f} отсчёта "
        f"({res.delay_frac_samples / cfg.FS * 1e9:+.0f} нс)",
        f"SNR по пилотам           : {res.snr_db:.1f} дБ",
        f"EVM данных               : {res.evm_db:.1f} дБ",
    ])
