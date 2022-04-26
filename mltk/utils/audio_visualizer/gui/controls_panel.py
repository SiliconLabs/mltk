
import os
import sys
import traceback
import types
import wx


from ..settings import SettingsManager
from ..audio_visualizer import AudioVisualizer

from . import res
from .generated.ControlsPanel import ControlsPanel
from .generated import get_widget
from .readme_dialog import ReadmeDialog



class VisualizerControlsPanel(ControlsPanel):
    
    def __init__(self, parent, id):
        ControlsPanel.__init__(self, parent, id)
        self.settings = SettingsManager.instance()
        self.visualizer = AudioVisualizer.instance()
        self.visualizer.spectrogram_canvas = get_widget('spectrogram_canvas')
        self.visualizer.update_gui_settings = self.load_settings
        self.reload_required = False
        self.replay_required = False
        self.debounce_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_debounce_timer, self.debounce_timer)
        
        for key, widget in self.settings.widgets.items():
            config = self.settings.get_config(key)
            
            if 'min' in config and 'max' in config:
                widget.SetRange(config['min'], config['max'])

            if 'inc' in config:
                widget.SetIncrement(config['inc'])
            
            widget.default_value = config['default']

        self.load_settings()
        
        
        def _display_readme(self, event):
            dlg = ReadmeDialog(self)
            try:
                dlg.ShowModal()
            finally:
                dlg.Destroy()

        frame = wx.GetTopLevelParent(self)
        frame.on_display_readme =  types.MethodType(_display_readme, frame)

        self._set_widgets_enabled('transform')
        self._set_widgets_enabled('frontend')
        self._load_audio_file()
        self._load_model_file()


    def load_settings(self):
        for key, widget in self.settings.widgets.items():
            widget.SetValue(self.settings.get(key))

        
    def on_txt_audio_file_path(self, event):
        self._load_audio_file(event.Value)


    def on_btn_find_audio_file(self, event):
        with wx.FileDialog(self, "Open WAV file", wildcard="WAV file (*.wav)|*.wav",
                              style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
        
            old_file = self.txtFilePath.GetValue()
            if old_file:
                fileDialog.SetDirectory(os.path.dirname(old_file))
            
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind
            
            # Proceed loading the file chosen by the user
            self._load_audio_file(fileDialog.GetPath())


    def on_btn_play_audio(self, event=None):
        try:
            self.visualizer.play_audio()
        except Exception as e:
            self._log_err(e, 'Failed to play audio')


    def on_txt_model_file_path(self, event):
        self._load_model_file(event.Value)


    def on_btn_find_model_file(self, event):
        with wx.FileDialog(self, "Open MLTK Model", wildcard="Python file (*.py)|*.py|MLTK Model Archive (*.mltk.zip)|*.mltk.zip",
                              style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as fileDialog:
        
            old_file = self.txtFilePath.GetValue()
            if old_file:
                fileDialog.SetDirectory(os.path.dirname(old_file))
            
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind
            
            # Proceed loading the file chosen by the user
            self._load_model_file(fileDialog.GetPath())


    def on_btn_load_model(self, event=None):
        self._load_model_file()


    def on_btn_step_forward(self, event):
        self.visualizer.step_direction(1)


    def on_btn_step_backward(self, event):
        self.visualizer.step_direction(-1)

    
    def on_setting_updated(self, event):
        if isinstance(event.EventObject, wx.CheckBox):
            v = True if event.Int else False
            
        elif isinstance(event.EventObject, wx.ComboBox):
            v = event.String
            
        elif hasattr(event, 'String'):
            try:
                v = float(event.String) 
            except:
                v = event.String
            
        else:
            v = event.Value
            
    

        if not self._check_other_settings(event.EventObject, v):
            return 
        
        self.settings.set(event.EventObject, v, save=False)
        self.replay_required = self.chkAutoPlay.IsChecked() and self.settings.is_in_group(event.EventObject, 'transform')
        self.reload_required = self.settings.is_in_group(event.EventObject, 'general')
            
        if self.btnPlay.Enabled:
            self.debounce_timer.StartOnce(500)
    
    
    def _check_other_settings(self, widget, value):
        key = self.settings.get_key(widget)
        if key == 'general.sample_rate':
            widget = self.settings.get_widget('frontend.upper_band_limit')
            upper_band_limit = widget.GetValue()
            max_limit = (value / 2)
            if upper_band_limit > max_limit:
                widget.SetValue(max_limit)
                self.settings.set('frontend.upper_band_limit', max_limit, save=False)
        
        elif key == 'frontend.upper_band_limit':
            sample_rate = self.settings.get('general.sample_rate')
            
            if sample_rate < value * 2:
                old_value = self.settings.get('frontend.upper_band_limit')
                widget.SetValue(old_value)
                return False
            
        elif key == 'frontend.window_size':
            widget = self.settings.get_widget('frontend.window_step')
            window_step = widget.GetValue()
            if window_step > value:
                widget.SetValue(value)
                self.settings.set('frontend.window_step', value, save=False)
        
        elif key == 'frontend.window_step':
            window_size = self.settings.get('frontend.window_size')
            
            if window_size < value:
                old_value = self.settings.get('frontend.window_step')
                widget.SetValue(old_value)
                return False
        
        
        return True


    def _on_debounce_timer(self, event):
        try:
            if self.reload_required:
                self._load_audio_file(self.txtFilePath.GetValue())
            
            self.settings.save()
        except Exception as e:
            self.settings.load()
            self._log_err(e, 'Failed to save: {}')
            return 
            
        self._draw()
        
        if self.replay_required:
            self.replay_required = False
            self.on_btn_play_audio()



    def on_chk_enable_transform(self, event):
        self._set_widgets_enabled('transform')
        self.settings.set('transform.enabled', self.chkEnableTransform.IsChecked(), save=True)
        self._draw()
        

    def on_chk_auto_play(self, event):
        self.settings.set('general.auto_play', self.chkAutoPlay.IsChecked(), save=True)


    def on_btn_reset_transform(self, event):
        self.visualizer.reset('transform')
        self._draw()
        for key, widget in self.settings.widgets.items():
            widget.SetValue(self.settings.get(key))


    def on_chk_enable_frontend(self, event):
        self._set_widgets_enabled('frontend')
        self.settings.set('frontend.enabled', self.chkEnableFrontend.IsChecked(), save=True)
        self._draw()


    def on_btn_reset_frontend(self, event):
        self.visualizer.reset('frontend')
        self._draw()
        for key, widget in self.settings.widgets.items():
            widget.SetValue(self.settings.get(key))


    def _load_audio_file(self, filepath=None):
        if filepath is None:
            filepath = self.txtFilePath.GetValue()
        
        self.reload_required = False 
        
        if not filepath or not os.path.exists(filepath):
            self.btnPlay.Disable()
            return
            
        try:
            self.visualizer.load_audio(filepath)
            self.settings.set('general.path', filepath, save=True)
            self.txtFilePath.SetValue(filepath)
            self.btnPlay.Enable()
        except Exception as e:
            self.btnPlay.Disable()
            self._log_err(e, 'Failed to load audio: {}')
            return False

        self._draw()
        
        return True


    def _load_model_file(self, model=None):
        if model is None:
            model = self.txtCmfModelPath.GetValue()
        self.reload_required = False 
        
        if not model or not os.path.exists(model):
            self.btnLoadModel.Disable()
            return

        try:
            self.visualizer.load_model(model)
            self.settings.set('general.model', model, save=True)
            self.txtCmfModelPath.SetValue(model)
            self.btnLoadModel.Enable()
        except Exception as e:
            self.btnLoadModel.Disable()
            self._log_err(e, 'Failed to load CMF model: {}')
            return False

        self._draw()
        
        return True

    def _draw(self):
        try:
            self.visualizer.draw() 
        except Exception as e:
            self._log_err(e, 'Failed to graph audio: {}')


    def _set_widgets_enabled(self, type):
        if type == 'transform':
            enabled = self.chkEnableTransform.IsChecked()
        else:
            enabled = self.chkEnableFrontend.IsChecked()
        settings = self.settings.get(type)
        
        for key in settings.keys():
            if key == 'enabled':
                continue 
            
            self.settings.get_widget(type + '.' + key).Enable(enabled)
        

    def _log_err(self, ex, msg):
        self.visualizer.logger.error(msg, exc_info=ex)
        wx.LogError('{}: {}'.format(msg, ex))

