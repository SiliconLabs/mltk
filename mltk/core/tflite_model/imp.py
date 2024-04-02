"""This is a work-around since the flatbuffer package 
still calls the imp package which is no longer available in python 3.12"""

from importlib.util import find_spec

def find_module(name:str):
    if not find_spec(name):
      raise ImportError