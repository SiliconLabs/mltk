import os
import inspect
import types
import collections
from typing import Dict

from mltk.utils.python import as_list


CallableType = (types.FunctionType,types.MethodType,types.LambdaType)
DictType = (dict, collections.defaultdict,collections.OrderedDict, collections.UserDict)


class _Attribute(object):
    """A specific model attribute"""
    def __init__(
        self, 
        key,
        value, 
        readonly=False, 
        dtype=None, 
        normalize=None,
        setter=None
    ):
        self.key = key
        self._is_set = False
        self.readonly = readonly
        self.normalize = normalize
        self.setter = setter
        
        if dtype is not None:
            dtype = tuple(as_list(dtype))
        self.dtype = dtype

        if value is not None:
            self.value = value

    @property
    def is_set(self):
        return self._is_set

    @property
    def value(self):
        if not self._is_set:
            raise Exception(f'No value set for {self.key}')
        return self._value
    @value.setter
    def value(self, v):
        self._is_set = True
        self._value = v



class MltkModelAttributes(object):
    """Container to hold the various attributes of a MltkModel"""
    def __init__(self):
        self._entries : Dict[str,_Attribute] = {}


    def register(
        self, 
        key: str, 
        value=None, 
        readonly=False, 
        dtype=None, 
        override=False, 
        normalize=None,
        setter=None
    ):
        """Register an attribute"""
        if not override and key in self._entries:
            raise Exception(f'Model attribute: {key} has already been registered')

        self._entries[key] = _Attribute(
            key, 
            value, 
            readonly=readonly, 
            dtype=dtype, 
            normalize=normalize, 
            setter=setter
        )


    def contains(self, key:str) -> bool:
        """Return if an attribute with the given keep has been previously registered"""
        key = self._resolve_key(key)
        return key in self._entries


    def get_value(self, key: str, **kwargs):
        """Return the value of the attribute with the given key"""
        key = self._resolve_key(key)
        if key not in self._entries:
            raise AttributeError(f'No model attribute found with name: {key}')

        if not self._entries[key].is_set and 'default' in kwargs:
            self._entries[key].value = kwargs['default']
        
        # If attribute value is callable and NOT a callable dtype, 
        # then the user provided a function to be called to generate the value,
        # so call the user function and return the value the function returns
        if callable(self._entries[key].value) and \
           (self._entries[key].dtype is None or len(set(self._entries[key].dtype) and set(CallableType)) == 0):
            return self._entries[key].value()
        
        return self._entries[key].value


    def set_value(self, key:str, value):
        """Set the value of an attribute with the given key"""
        key = self._resolve_key(key)
        if key not in self._entries:
            raise AttributeError(f'Model attribute: {key} has not been previously registered')
        
        if self._entries[key].readonly:
            raise AttributeError(f'Model attribute: {key} is read-only')

        if value is not None:
            if self._entries[key].dtype is not None:
                if not callable(value) and not isinstance(value, self._entries[key].dtype):
                    s = ', '.join(f'{x.__name__}' for x in self._entries[key].dtype)
                    raise AttributeError(f'Model attribute: {key} must be a dtype of {s} or a callable function')
            
            if not callable(value) and self._entries[key].normalize is not None:
                value = self._entries[key].normalize(value)

        if self._entries[key].setter is not None:
            self._entries[key].setter(value)
        else:
            self._entries[key].value = value


    def value_is_set(self, key:str):
        """Return if the value of the attribute with the given key has been previously set"""
        key = self._resolve_key(key)
        if key not in self._entries:
            raise AttributeError(f'Model attribute: {key} has not been previously registered')
        return self._entries[key].is_set


    def _resolve_key(self, key:str) -> str:
        if key.startswith('*'):
            for name in self._entries:
                if name.endswith(key[1:]):
                    return name
        elif key.endswith('*'):
            for name in self._entries:
                if name.startswith(key[:-1]):
                    return name
        else:
            return key 
        
        raise AttributeError(f'No model attribute found with wildcard entry: {key}')


    def __getitem__(self, key):
        return self.get_value(key)

        
    def __setitem__(self, key, value):
        self.set_value(key, value)


    def __contains__(self, key):
        return self.contains(key)



def MltkModelAttributesDecorator():
    """Class decorator that automatically registers model mixin attributes 
    before any class properities or public functions are accessed
    """
    def decorate(cls):
        for name, val in inspect.getmembers(cls):
            if name.startswith('_'):
                continue 
            if isinstance(val, (types.MethodType, types.FunctionType)) and os.environ.get('MLTK_BUILD_DOCS') != '1':
                setattr(cls, name, _check_attributes_registered_decorator(val))
            elif isinstance(val, property):
                setattr(cls, name, _decorate_property(val))
        return cls
    return decorate


def _check_attributes_registered_decorator(f):
    """Decorator that is applied to model mixin properties
    and public functions to ensure model attributes are automatically
    registered before a property or public function is accessed
    """
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, '_attributes_are_registered'):
            self._attributes_are_registered = True # pylint: disable=protected-access
            called_funcs = []
            post_registration_callbacks = []
            inherited_classes = self.__class__.mro()
            for cls in inherited_classes:
                register_attributes_func = getattr(cls, '_register_attributes', None)
                if register_attributes_func and register_attributes_func not in called_funcs:
                    called_funcs.append(register_attributes_func)
                    post_registration_callback = register_attributes_func(self)
                    if post_registration_callback:
                        post_registration_callbacks.append(post_registration_callback)
            
            for post_registration_callback in post_registration_callbacks:
                post_registration_callback()
        
        return f(self, *args, **kwargs)
    return wrapper

def _decorate_property(prop: property):
    """Update a class property to automatically register model mixin 
    attributes before the class property is access"""
    getx = None 
    setx = None
    if prop.fget is not None:
        getx =  _check_attributes_registered_decorator(prop.fget)
    if prop.fset is not None:
        setx = _check_attributes_registered_decorator(prop.fset)

    return property(getx, setx, prop.fdel, prop.__doc__)