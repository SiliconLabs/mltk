# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name
"""MobileNet v2 models for Keras.

MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNetV2 is very similar to the original MobileNet,
except that it uses inverted residual blocks with
bottlenecking features. It has a drastically lower
parameter count than the original MobileNet.
MobileNets support any input size greater
than 32 x 32, with larger image sizes
offering better performance.

The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and ``alpha`` parameter,
all 22 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using ``alpha`` values of
1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4
For each of these ``alph`a` values, weights for 5 different input image sizes
are provided (224, 192, 160, 128, and 96).


.. seealso::
   `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`_
"""

from tensorflow.keras.models import Model
from tensorflow.keras import layers 
from tensorflow.keras import backend


def MobileNetV2(
  input_shape,
  alpha=0.35,
  include_top=True,
  pooling=None,
  classes=1000,
  classifier_activation='softmax',
  last_block_filters=None,
  **kwargs
):
  """Instantiates the MobileNetV2 architecture.

  .. seealso::
     * `MobileNetV2: Inverted Residuals and Linear Bottlenecks <https://arxiv.org/abs/1801.04381>`_

  Optionally loads weights pre-trained on ImageNet.

  Note: each Keras Application expects a specific kind of input preprocessing.
  For MobileNetV2, call ``tf.keras.applications.mobilenet_v2.preprocess_input``
  on your inputs before passing them to the model.

  Arguments:
    input_shape: shape tuple, to be specified if you would
      like to use a model with an input image resolution that is not
      (224, 224, 3).
      It should have exactly 3 inputs channels (224, 224, 3).
    alpha: Float between 0 and 1. controls the width of the network.
      This is known as the width multiplier in the MobileNetV2 paper,
      but the name is kept for consistency with ``applications.MobileNetV1``
      model in Keras.

      - If ``alpha`` < 1.0, proportionally decreases the number
          of filters in each layer.
      - If ``alpha`` > 1.0, proportionally increases the number
          of filters in each layer.
      - If ``alpha`` = 1, default number of filters from the paper
          are used at each layer.
      
    include_top: Boolean, whether to include the fully-connected
      layer at the top of the network. Defaults to ``True``.
    pooling: String, optional pooling mode for feature extraction
      when ``include_top`` is ``False``.
      
      - ``None`` means that the output of the model
          will be the 4D tensor output of the
          last convolutional block.
      - ``avg`` means that global average pooling
          will be applied to the output of the
          last convolutional block, and thus
          the output of the model will be a
          2D tensor.
      - ``max`` means that global max pooling will
          be applied.
      
    classes: Integer, optional number of classes to classify images
      into, only to be specified if ``include_top`` is True, and
      if no ``weights`` argument is specified.
    classifier_activation: A ``str`` or callable. The activation function to use
      on the "top" layer. Ignored unless ``include_top=True``. Set
      ``classifier_activation=None`` to return the logits of the "top" layer.
    last_block_filters: The number of filters to use in the last block of the model.
     If omitted, default to 1280 which the standard model uses. Due to hardware constraints,
     this value must be decreased  (< 1024) to be fully optimized by the MVP hardare.
    **kwargs: For backwards compatibility only.

  Returns:
    A `keras.Model` instance.

  Raises:
    ValueError: in case of invalid argument for ``weights``,
      or invalid input shape or invalid alpha, rows when
      weights='imagenet'
    ValueError: if ``classifier_activation`` is not ``softmax`` or ``None`` when
      using a pretrained top layer.
  """

 
  img_input = layers.Input(shape=input_shape)
  channel_axis =  -1


  img_input = layers.Input(shape=input_shape)
  rows = input_shape[0]

  first_block_filters = _make_divisible(32 * alpha, 8)
  x = layers.Conv2D(
      first_block_filters,
      kernel_size=3,
      strides=(2, 2),
      padding='same',
      use_bias=False,
      name='Conv1')(img_input)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3, momentum=0.999, name='bn_Conv1')(
          x)
  x = layers.ReLU(6., name='Conv1_relu')(x)

  x = _inverted_res_block(
      x, filters=16, alpha=alpha, stride=1, expansion=1, block_id=0)

  x = _inverted_res_block(
      x, filters=24, alpha=alpha, stride=2, expansion=6, block_id=1)
  x = _inverted_res_block(
      x, filters=24, alpha=alpha, stride=1, expansion=6, block_id=2)

  x = _inverted_res_block(
      x, filters=32, alpha=alpha, stride=2, expansion=6, block_id=3)
  x = _inverted_res_block(
      x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=4)
  x = _inverted_res_block(
      x, filters=32, alpha=alpha, stride=1, expansion=6, block_id=5)

  x = _inverted_res_block(
      x, filters=64, alpha=alpha, stride=2, expansion=6, block_id=6)
  x = _inverted_res_block(
      x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=7)
  x = _inverted_res_block(
      x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=8)
  x = _inverted_res_block(
      x, filters=64, alpha=alpha, stride=1, expansion=6, block_id=9)

  x = _inverted_res_block(
      x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=10)
  x = _inverted_res_block(
      x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=11)
  x = _inverted_res_block(
      x, filters=96, alpha=alpha, stride=1, expansion=6, block_id=12)

  x = _inverted_res_block(
      x, filters=160, alpha=alpha, stride=2, expansion=6, block_id=13)
  x = _inverted_res_block(
      x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=14)
  x = _inverted_res_block(
      x, filters=160, alpha=alpha, stride=1, expansion=6, block_id=15)

  x = _inverted_res_block(
      x, filters=320, alpha=alpha, stride=1, expansion=6, block_id=16)

  # no alpha applied to last conv as stated in the paper:
  # if the width multiplier is greater than 1 we
  # increase the number of output channels
  if last_block_filters is None:
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

  x = layers.Conv2D(
      last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis, epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(
          x)
  x = layers.ReLU(6., name='out_relu')(x)

  if include_top:
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)

  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  inputs = img_input

  # Create model.
  model = Model(inputs, x, name=f'mobilenetv2_%0.2f_%s' % (alpha, rows))

  return model


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
  """Inverted ResNet block."""
  channel_axis = -1

  in_channels = backend.int_shape(inputs)[channel_axis]
  pointwise_conv_filters = int(filters * alpha)
  pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
  x = inputs
  prefix = 'block_{}_'.format(block_id)

  if block_id:
    # Expand
    x = layers.Conv2D(
        expansion * in_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'expand')(
            x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand_BN')(
            x)
    x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
  else:
    prefix = 'expanded_conv_'

  # Depthwise
  if stride == 2:
    x = layers.ZeroPadding2D(
        padding=correct_pad(x, 3),
        name=prefix + 'pad')(x)
  x = layers.DepthwiseConv2D(
      kernel_size=3,
      strides=stride,
      activation=None,
      use_bias=False,
      padding='same' if stride == 1 else 'valid',
      name=prefix + 'depthwise')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'depthwise_BN')(
          x)

  x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

  # Project
  x = layers.Conv2D(
      pointwise_filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      activation=None,
      name=prefix + 'project')(
          x)
  x = layers.BatchNormalization(
      axis=channel_axis,
      epsilon=1e-3,
      momentum=0.999,
      name=prefix + 'project_BN')(
          x)

  if in_channels == pointwise_filters and stride == 1:
    return layers.Add(name=prefix + 'add')([inputs, x])
  return x


def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v

def correct_pad(inputs, kernel_size):
  """Returns a tuple for zero-padding for 2D convolution with downsampling.

  Args:
    inputs: Input tensor.
    kernel_size: An integer or tuple/list of 2 integers.

  Returns:
    A tuple.
  """
  img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
  input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]
  if isinstance(kernel_size, int):
    kernel_size = (kernel_size, kernel_size)
  if input_size[0] is None:
    adjust = (1, 1)
  else:
    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
  correct = (kernel_size[0] // 2, kernel_size[1] // 2)
  return ((correct[0] - adjust[0], correct[0]),
          (correct[1] - adjust[1], correct[1]))