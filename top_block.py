#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: FM Radio PlutoSDR
# Author: Jay Patel
# Copyright: 2020
# Description: FM Radio PlutoSDR
# GNU Radio version: 3.10.11.0

from PyQt5 import Qt
from gnuradio import qtgui
from PyQt5 import QtCore
from gnuradio import analog
from gnuradio import audio
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import iio
import sip
import threading



class top_block(gr.top_block, Qt.QWidget):

    def __init__(self, audio_device='default', decimation=1, uri='ip:pluto.local'):
        gr.top_block.__init__(self, "FM Radio PlutoSDR", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("FM Radio PlutoSDR")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "top_block")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Parameters
        ##################################################
        self.audio_device = audio_device
        self.decimation = decimation
        self.uri = uri

        ##################################################
        # Variables
        ##################################################
        self.sample_rate = sample_rate = 2800000
        self.fm_station = fm_station = int(95.2e6)

        ##################################################
        # Blocks
        ##################################################

        self._fm_station_range = qtgui.Range(int(92e6), int(106e6), int(10000), int(95.2e6), 200)
        self._fm_station_win = qtgui.RangeWidget(self._fm_station_range, self.set_fm_station, "station", "counter_slider", int, QtCore.Qt.Horizontal)
        self.top_layout.addWidget(self._fm_station_win)
        self.qtgui_sink_x_0 = qtgui.sink_c(
            1024, #fftsize
            window.WIN_BLACKMAN_hARRIS, #wintype
            fm_station, #fc
            sample_rate, #bw
            'Receive Signal', #name
            True, #plotfreq
            True, #plotwaterfall
            True, #plottime
            True, #plotconst
            None # parent
        )
        self.qtgui_sink_x_0.set_update_time(1.0/10)
        self._qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)

        self.qtgui_sink_x_0.enable_rf_freq(False)

        self.top_layout.addWidget(self._qtgui_sink_x_0_win)
        self.low_pass_filter_0 = filter.fir_filter_ccf(
            (sample_rate // (384000 * decimation)),
            firdes.low_pass(
                1,
                (int(sample_rate / decimation)),
                44100,
                44100,
                window.WIN_HAMMING,
                6.76))
        self.iio_pluto_source_0 = iio.fmcomms2_source_fc32('192.168.2.1' if '192.168.2.1' else iio.get_pluto_uri(), [True, True], 0x20000)
        self.iio_pluto_source_0.set_len_tag_key('packet_len')
        self.iio_pluto_source_0.set_frequency(fm_station)
        self.iio_pluto_source_0.set_samplerate(sample_rate)
        self.iio_pluto_source_0.set_gain_mode(0, 'manual')
        self.iio_pluto_source_0.set_gain(0, 64)
        self.iio_pluto_source_0.set_quadrature(True)
        self.iio_pluto_source_0.set_rfdc(True)
        self.iio_pluto_source_0.set_bbdc(True)
        self.iio_pluto_source_0.set_filter_params('Auto', '', 0, 0)
        self.audio_sink_0 = audio.sink(48000, audio_device, True)
        self.analog_wfm_rcv_0 = analog.wfm_rcv(
        	quad_rate=384000,
        	audio_decimation=8,
        )


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_wfm_rcv_0, 0), (self.audio_sink_0, 0))
        self.connect((self.iio_pluto_source_0, 0), (self.low_pass_filter_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.analog_wfm_rcv_0, 0))
        self.connect((self.low_pass_filter_0, 0), (self.qtgui_sink_x_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "top_block")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_audio_device(self):
        return self.audio_device

    def set_audio_device(self, audio_device):
        self.audio_device = audio_device

    def get_decimation(self):
        return self.decimation

    def set_decimation(self, decimation):
        self.decimation = decimation
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, (int(self.sample_rate / self.decimation)), 44100, 44100, window.WIN_HAMMING, 6.76))

    def get_uri(self):
        return self.uri

    def set_uri(self, uri):
        self.uri = uri

    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate
        self.iio_pluto_source_0.set_samplerate(self.sample_rate)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, (int(self.sample_rate / self.decimation)), 44100, 44100, window.WIN_HAMMING, 6.76))
        self.qtgui_sink_x_0.set_frequency_range(self.fm_station, self.sample_rate)

    def get_fm_station(self):
        return self.fm_station

    def set_fm_station(self, fm_station):
        self.fm_station = fm_station
        self.iio_pluto_source_0.set_frequency(self.fm_station)
        self.qtgui_sink_x_0.set_frequency_range(self.fm_station, self.sample_rate)



def argument_parser():
    description = 'FM Radio PlutoSDR'
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "--audio-device", dest="audio_device", type=str, default='default',
        help="Set Audio device [default=%(default)r]")
    parser.add_argument(
        "--decimation", dest="decimation", type=intx, default=1,
        help="Set Decimation [default=%(default)r]")
    parser.add_argument(
        "--uri", dest="uri", type=str, default='ip:pluto.local',
        help="Set URI [default=%(default)r]")
    return parser


def main(top_block_cls=top_block, options=None):
    if options is None:
        options = argument_parser().parse_args()
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        gr.logger("realtime").warn("Error: failed to enable real-time scheduling.")

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls(audio_device=options.audio_device, decimation=options.decimation, uri=options.uri)

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
