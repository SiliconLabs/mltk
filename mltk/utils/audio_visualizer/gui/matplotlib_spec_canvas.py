#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# a class to make matplotib.backend_wxagg.FigureCanvasWxAgg compatible to wxGlade

import threading 
import wx
import numpy as np

from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import librosa
import librosa.display




class MatplotlibSpectrogramCanvas(FigureCanvas):
    def __init__(self, parent, id=wx.ID_ANY, **fig_kwargs):
        # 1x1 grid, first subplot
        self.figure = Figure(**fig_kwargs)
        self.figure.set_tight_layout({"pad": .0}) 
        self.axes = self.figure.add_subplot(111)
        FigureCanvas.__init__(self, parent, id, self.figure)


    def clear(self, draw=True):
        self.axes.clear()
        if draw:
            self.draw()
        
    def plot(
        self, 
        data, 
        sr, 
        window_size_ms, 
        hop_length_ms, 
        n_mels, 
        fmin=None, 
        fmax=None, 
        precomputed=False, 
        axis_labels=True, 
        size_label=True
    ):
        self.axes.clear()
        hop_length = int((hop_length_ms / 1000) * sr)
        
        if not precomputed:
            window_size = int((window_size_ms / 1000) * sr)
            n_fft = 1 
            
            while n_fft < window_size:
                n_fft <<= 1
            
            
            S = librosa.feature.melspectrogram(
                y=data, 
                sr=sr, 
                n_fft=n_fft, 
                win_length=window_size,
                hop_length=hop_length, 
                n_mels=n_mels, 
                fmin=fmin, 
                fmax=fmax,
                center=False
            )
            data = librosa.power_to_db(S)
            v_max = 20
            v_min =-90
        else:
            data = data.transpose()
            v_max = 1000
            v_min = 100

        y_axis = 'off'
        x_axis = 'off'
        if axis_labels:
            y_axis='mel'
            x_axis='time'
            
        librosa.display.specshow(
            data, 
            ax=self.axes, 
            sr=sr, 
            hop_length=hop_length,
            fmin=fmin, 
            fmax=fmax, 
            y_axis=y_axis, 
            x_axis=x_axis,
            vmax = v_max,
            vmin = v_min
        )

        lbl = '{}x{}'.format(*data.shape) if size_label else ''
        self.axes.text(.01, 0.97, lbl, 
            horizontalalignment='left',
            verticalalignment='center', 
            color='white',
            fontproperties=FontProperties(weight='heavy', size='x-large'),
            transform=self.axes.transAxes
        )

        draw_event = threading.Event()

        def _draw_spectrogram(self, draw_event):
            self.draw()
            draw_event.set()

        try:
            wx.CallAfter(_draw_spectrogram, self, draw_event)
            draw_event.wait()
        except:
            pass
