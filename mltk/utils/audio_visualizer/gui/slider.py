

import wx

from ..settings import SettingsManager



class Slider(wx.Panel):
    
    
    def __init__(self, parent, id, widget, key, dtype='float', digits=2):
        wx.Panel.__init__(self, parent, id)
        self.widget = widget
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.dtype = dtype
        self.btnReset = wx.Button(self, wx.ID_ANY, label='X', style=wx.BU_EXACTFIT)
        self.Bind(wx.EVT_BUTTON, self._on_btn_reset, self.btnReset)
         
        
        if dtype == 'float':
            self.spinner = wx.SpinCtrlDouble(self, wx.ID_ANY, size=(75, -1), style=wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER)
            self.spinner.SetDigits(digits)
            self.Bind(wx.EVT_SPINCTRLDOUBLE, self._on_spinner_updated, self.spinner)
            self.scale = digits*10.0
            self._to_spinner_value = self._to_float
        else:
            self.spinner = wx.SpinCtrl(self, wx.ID_ANY, size=(75, -1), style=wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER)
            self.Bind(wx.EVT_SPINCTRL, self._on_spinner_updated, self.spinner)
            self.scale = 1.0
            self._to_spinner_value = self._to_int
       
        self.Bind(wx.EVT_TEXT_ENTER, self._on_spinner_updated, self.spinner)
        
        self.slider = wx.Slider(self, wx.ID_ANY)
        
        self.Bind(wx.EVT_SLIDER, self._on_slider_updated, self.slider)
        
        sizer.Add(self.btnReset, 0, wx.RIGHT | wx.EXPAND, 3)
        sizer.Add(self.spinner, 0, wx.ALIGN_CENTER_VERTICAL)
        sizer.Add(self.slider, 1, wx.LEFT | wx.EXPAND, 3)
        
        self.SetSizer(sizer)
        
        SettingsManager.register_widget(key, self)


    def SetMin(self, v):
        self.spinner.SetMin(self._to_spinner_value(v))
        self.slider.SetMin(self._to_int(v))
    
    
    def SetMax(self, v):
        self.spinner.SetMax(self._to_spinner_value(v))
        self.slider.SetMax(self._to_int(v))
    
    
    def SetRange(self, min, max):
        self.spinner.SetRange(self._to_spinner_value(min), self._to_spinner_value(max))
        self.slider.SetRange(self._to_int(min), self._to_int(max))
    
    
    def SetIncrement(self, v):
        if isinstance(self.spinner, wx.SpinCtrlDouble):
            self.spinner.SetIncrement(self._to_spinner_value(v))
        
    
    def SetValue(self, v):
        self.spinner.SetValue(self._to_spinner_value(v)) 
        self.slider.SetValue(self._to_int(v))
    
    
    def GetValue(self):
        if self.dtype == 'float':
            return self.spinner.GetValue() 
        else:
            return int(self.slider.GetValue())
    
    
    def _on_spinner_updated(self, event):
        if hasattr(event, 'Value'):
            v = event.Value 
        if hasattr(event, 'Int'):
            v = event.Int 
        else:
            v = float(event.String)
        
        self.slider.SetValue(self._to_int(v))
        self._on_event()

    
    def _on_slider_updated(self, event):
        v = event.Int 
        v = self._to_spinner_value(v)
        self.spinner.SetValue(v)  
        self._on_event()
        
        
    def _on_event(self):
        class E(object):
            Value = None 
            EventObject = None 
        
        e = E()
        e.Value = self.GetValue()
        e.EventObject = self
        self.widget.on_setting_updated(e)
        

    def _on_btn_reset(self, event):
        self.SetValue(self.default_value)
        self._on_event()


    def _to_int(self, v):
        if isinstance(v, int):
            return v 
        
        return int(v * self.scale)
    
    
    def _to_float(self, v):
        if isinstance(v, float):
            return v 
        
        return v / self.scale
    