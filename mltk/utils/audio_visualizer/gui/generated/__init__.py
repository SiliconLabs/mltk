


_widgets = {}


def add_widget(name, widget):
    global _widgets
    _widgets[name] = widget 
    
    
def get_widget(name):
    return _widgets[name]