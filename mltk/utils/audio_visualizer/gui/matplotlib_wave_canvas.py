
import wx


from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import librosa
import librosa.display



class MatplotlibWavePlotCanvas(FigureCanvas):
    def __init__(self, parent, id=wx.ID_ANY):
        # 1x1 grid, first subplot
        self.figure = Figure(figsize=(6,2))
        self.axes = self.figure.add_subplot(111)
        FigureCanvas.__init__(self, parent, id, self.figure)
        

    def clear(self, draw=True):
        self.axes.clear()
        if draw:
            self.draw()

    
    def plot(self, audio, sample_rate):
        self.axes.clear()
        librosa.display.waveplot(audio, sample_rate, ax=self.axes)
        wx.CallAfter(lambda: self.draw())

