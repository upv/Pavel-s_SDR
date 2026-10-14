"""5G NR PUSCH over the air with the ANTSDR E316: the WirelessNN waveform, received for real.

    wirelessnn   locate ~/Sources/WirelessNN, build its SimConfig / PUSCHTransmitter / PuschRx
    waveform     one PUSCH slot (Sionna OFDM modulator) + gap -> frame; DMRS reference symbol
    sync         frame detection, timing and CFO (CP metric + DMRS correlation), noise from the gap
    sdr          libiio access to the E316 (TX cyclic buffer, RX capture with stale-block flush)
    rx_pusch     CLI receiver: capture / file / simulation -> sync -> OFDM demod -> CE -> LMMSE -> LDPC
    tx_pusch     CLI transmitter: generate the frame and play it cyclically on TX1
"""
