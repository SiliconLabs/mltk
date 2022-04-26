

import wx 
import wx.html
import markdown
from . import res



class ReadmeDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, wx.ID_ANY, 'Read Me', size=(800,800), style=wx.RESIZE_BORDER|wx.DEFAULT_DIALOG_STYLE)
        _icon = wx.NullIcon
        _icon.CopyFromBitmap(wx.Bitmap(res.path('gui/favicon.ico'), wx.BITMAP_TYPE_ANY))
        self.SetIcon(_icon)
        
        panel = ReadmePanel(self)
        

class ReadmePanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        
        with open(res.path('README.md'), 'r') as fp:
            md = fp.read()
            
        win = wx.html.HtmlWindow(self)
        html = markdown.markdown(md)
        win.SetPage(html)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(win, 1, wx.ALL|wx.EXPAND)
        self.SetSizer(sizer)
        
