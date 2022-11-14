
try:
    import wx
except:
    raise RuntimeError('Failed import wx Python package, try running: pip install wxpython')

from .gui import res
from .gui.generated.VisualizerFrame import VisualizerFrame


class VisualizerApp(wx.App):
    def __init__(self):
        wx.App.__init__(self, 0)
        
    def OnInit(self):
        self.frame = VisualizerFrame(None, wx.ID_ANY, "")
        _icon = wx.NullIcon
        _icon.CopyFromBitmap(wx.Bitmap(res.path('gui/favicon.ico'), wx.BITMAP_TYPE_ANY))
        self.frame.SetIcon(_icon)
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True
    
