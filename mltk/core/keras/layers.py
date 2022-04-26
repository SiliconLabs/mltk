import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



#define custom layer with weight reuse
class Conv2DTranspose_tied(keras.layers.Layer):
    def __init__(self, layer, activation=None, **kwargs):
        self.layer = layer
        if 'kernel_shape' in kwargs:
            self.kernel_shape = kwargs['kernel_shape']
            self.layer_input_shape = kwargs['layer_input_shape']
            del kwargs['kernel_shape']
            del kwargs['layer_input_shape']
        else:
            self.kernel_shape = tuple(self.layer.kernel.shape)
            self.layer_input_shape = tuple(self.layer.input_shape)
        self.activation = keras.activations.get(activation)
        super().__init__(trainable=False, **kwargs)


    def call(self, inputs, training=None, mask=None):
        if self.layer.built:
            kernel = self.layer.kernel
        else:
            kernel = np.empty(self.kernel_shape, dtype=self.layer.dtype)

        output = K.conv2d_transpose(inputs,
                          kernel=kernel,
                          output_shape=self.layer_input_shape,
                          strides=self.layer.strides,
                          padding=self.layer.padding)
        return self.activation(output)


    def get_config(self):
        return {
            'layer': self.layer,
            'activation': self.activation,
            'kernel_shape': self.kernel_shape,
            'layer_input_shape': self.layer_input_shape,
        }

    @property
    def trainable_weights(self):
        # This layer is tied to another layer and thus doesn't have any weights
        return []

    @property
    def non_trainable_weights(self):
        # This layer is tied to another layer and thus doesn't have any weights
        return []


class DenseTranspose_tied(keras.layers.Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense

        if 'dense_weights_shape' in kwargs:
            self.dense_weights_shape = kwargs['dense_weights_shape']
            del kwargs['dense_weights_shape']
        else:
            self.dense_weights_shape = tuple(self.dense.weights[0].shape)

        self.activation = keras.activations.get(activation)
        super().__init__(trainable=False, **kwargs)
        

    def call(self, inputs, training=None, mask=None):
        if self.dense.built:
            weights = self.dense.weights[0] 
        else:
            weights = np.empty(self.dense_weights_shape, dtype=self.dense.dtype)

        z = tf.matmul(a=inputs, b=weights, transpose_b=True)

        return self.activation(z)


    def get_config(self):
        return {
            'activation': self.activation,
            'dense': self.dense,
            'dense_weights_shape': self.dense_weights_shape
        }

    @property
    def trainable_weights(self):
        # This layer is tied to another layer and thus doesn't have any weights
        return []

    @property
    def non_trainable_weights(self):
        # This layer is tied to another layer and thus doesn't have any weights
        return []
