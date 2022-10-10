from mltk.core.tflite_model.tflite_schema import BuiltinOperator
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

class TFOpLambda(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('TFOpLambda',))
        
    def process(self, layer):
        pass

class GlobalAveragePool2D(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('GlobalAveragePooling2D', BuiltinOperator.MEAN))
        
    def process(self, layer):
        pass

class GlobalMaxPool2D(Layer):
    
    def __init__(self):
        Layer.__init__(self, ('GlobalMaxPool2D', BuiltinOperator.REDUCE_MAX))
        
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
TFOpLambda()
GlobalAveragePool2D()
GlobalMaxPool2D()