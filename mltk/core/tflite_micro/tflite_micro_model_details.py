from typing import List
from mltk.utils.string_formatting import  format_units

from .tflite_micro_memory_plan import TfliteMicroMemoryPlan


class TfliteMicroModelDetails:
    """TF-Lite Micro Model Details"""

    def __init__(self, wrapper_details:dict):
        self._details:dict = wrapper_details
        self._memory_plan:TfliteMicroMemoryPlan = None

    @property
    def name(self) -> str:
        """Name of model"""
        return self._details['name']
    @property
    def version(self)-> int:
        """Version of model"""
        return self._details['version']
    @property
    def date(self)-> str:
        """Date of model in ISO8601 format"""
        return self._details['date']
    @property
    def description(self)-> str:
        """Description of model"""
        return self._details['description']
    @property
    def classes(self) -> List[str]:
        """List of class labels"""
        return self._details['classes']
    @property
    def hash(self)-> str:
        """Unique hash of model data"""
        return self._details['hash']
    @property
    def accelerator(self)-> str:
        """Accelerater kernels loaded into TFLM interpreter"""
        return self._details['accelerator']
    @property
    def runtime_memory_size(self)-> int:
        """Total amount of RAM required at runtime to run model"""
        return self._details['runtime_memory_size']

    @property
    def memory_plan(self) -> TfliteMicroMemoryPlan:
        """The generated tensor buffer layout used for this model"""
        return self._memory_plan


    def __str__(self):
        s = ''
        s += f"Name: {self.name}\n"
        s += f"Version: {self.version}\n"
        s += f"Date: {self.date}\n"
        s += f"Description: {self.description}\n"
        s += f"Hash: {self.hash}\n"
        s += f"Accelerator: {self.accelerator}\n"
        s += f"Classes: {', '.join(self.classes)}\n"
        s += f"Total runtime memory: {format_units(self.runtime_memory_size)}Bytes\n"
        return s
