
import os
import shutil
import yaml
import threading
import copy


from mltk.utils.python import merge_dict
from mltk.utils.path import create_user_dir



_settings_manager = None






class SettingsManager(object):
    
    def __init__(self):
        self.widgets = {}
        self.settings = {}
        self.config = {}
        self.lock = threading.RLock()
        self.saved_backup = False 
        self.settings_path = create_user_dir() + '/audio_visualizer_settings.yaml'
        
        
    @staticmethod
    def instance():
        global _settings_manager
        if _settings_manager is None:
            _settings_manager = SettingsManager() 
        return _settings_manager
            
    
    @staticmethod
    def register_widget(key, widget):
        instance = SettingsManager.instance()
        instance.widgets[key] = widget


    def save(self):
        with self.lock:
            if not self.saved_backup and os.path.exists(self.settings_path):
                self.saved_backup = True 
                shutil.copy(self.settings_path, self.settings_path + '.bak')
                
            os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
            with open(self.settings_path, 'w') as fp:
                yaml.dump(self.settings, fp, yaml.Dumper)

    def load(self):
        config_path = f'{os.path.dirname(os.path.abspath(__file__))}/config.yaml'
        with open(config_path, 'r') as fp:
            self.config = yaml.load(fp, yaml.Loader)

        self.load_default()
        
        try:
            with open(self.settings_path, 'r') as fp:
                loaded_settings = yaml.load(fp, yaml.Loader)
                merge_dict(self.settings, loaded_settings)
        except:
            pass
    
    
    def load_default(self):
        self.settings = {}
        for group, value in self.config.items():
            self.settings[group] = {}
            for key, value in value.items():
                self.settings[group][key] = value['default']


    def split_key(self, key_or_widget):
        key = None 
        
        if not isinstance(key_or_widget, str):
            for k, widget in self.widgets.items():
                if widget == key_or_widget:
                    key = k
                    break 
                
            if not key:
                raise Exception('Unknown widget')
            
        else:
            key = key_or_widget
            
        toks = key.split('.')
        group = toks[0]
        
        if not group in self.settings:
            raise Exception(f'No setting with group: {group} found')
        
        if len(toks) > 1:
            key = toks[1]
            
            if not key in self.settings[group]:
                raise Exception(f'Setting group: {group} does not have key: {key}')
        
        else:
            key = None
        
        return group, key


    def set(self, key, value, save=False):
        with self.lock:
            group, key = self.split_key(key)
            if not key:
                raise Exception('Must provide group and key')
            
            self.settings[group][key] = value
        
            if save:
                self.save()


    def get(self, key):
        with self.lock:
            group, key = self.split_key(key)
            
            if key:
                return self.settings[group][key]
            
            else:
                return self.settings[group]


    def get_snapshot(self, key=None):
        with self.lock:
            if key:
                retval = self.get(key)
            else:
                retval = self.settings
            return copy.deepcopy(retval)


    def get_config(self, key):
        group, key = self.split_key(key)
        return self.config[group][key]
    
    
    def get_widget(self, key):
        if key in self.widgets:
            return self.widgets[key]
        return None


    def get_key(self, key_or_widget):
        if isinstance(key_or_widget, str):
            return key_or_widget
        
        for key, widget in self.widgets.items():
            if widget == key_or_widget:
                return key 
            
        raise Exception('Widget not found')
        

    def is_in_group(self, key_or_widget, group):
        actual_group, _ = self.split_key(key_or_widget)
        return actual_group == group 
    