from tensorflow_lite_support.metadata.schema_py_generated import BuiltinOperator
from .layer import Layer


class Input(Layer):
    
    def __init__(self):
        Layer.__init__(self,'InputLayer')
        
    def process(self, layer):
        pass


class Flatten(Layer):
    
    def __init__(self):
        Layer.__init__(self, 'Flatten')
        
    def process(self, layer):
        pass
    

class Reshape(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Reshape', BuiltinOperator.RESHAPE))
        
    def process(self, layer):
        pass


class BatchNormalization(Layer):
    
    def __init__(self):
        Layer.__init__(self, 'BatchNormalization')
        
    def process(self, layer):
        pass

class Dropout(Layer):
    
    def __init__(self):
        Layer.__init__(self, 'Dropout')
        
    def process(self, layer):
        pass



class Split(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Split', BuiltinOperator.SPLIT))
        
    def process(self, layer):
        pass

class Concatenate(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('Concatenate', BuiltinOperator.CONCATENATION))
        
    def process(self, layer):
        pass

class ZeroPadding2D(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('ZeroPadding2D', BuiltinOperator.PAD))
        
    def process(self, layer):
        pass



Input()
Flatten()
Reshape()
BatchNormalization()
Dropout()
Split()
Concatenate()
ZeroPadding2D()
