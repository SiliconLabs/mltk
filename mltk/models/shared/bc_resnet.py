"""Model based on Broadcasted Residual Learning for Efficient Keyword Spotting.
model version based on paper:
https://arxiv.org/pdf/2106.04140.pdf
"""
import tensorflow as tf

def BCResNet(
    input_shape,
    label_count,
    first_filters=16,
    last_filters=32,
    blocks_n=[2, 2, 4, 4],
    filters=[8, 12, 16, 20],
    dilations=[(1,1), (2,1), (4,1), (8,1)],
    strides=[(1,1), (1,2), (1,2), (1,1)],
    dropouts=[0.1, 0.1, 0.1, 0.1],
    pools=[1, 1, 1, 1],
    paddings='same',
    max_pool=False,
    return_softmax=True
) -> tf.keras.Model: 
  if isinstance(input_shape, (list,tuple)):
    input_audio = tf.keras.layers.Input(shape=input_shape)
  else:
    input_audio = input_shape
  
  net = tf.keras.layers.Conv2D(filters=first_filters, kernel_size=5, strides=(1,2), padding=paddings)(input_audio)

  for n, n_filters, dilation, stride, dropout, pool in zip(blocks_n, filters, dilations, strides, dropouts, pools):
    # transition block:

    # 1x1 conv 
    net = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=1, strides=stride, padding='valid',use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.activations.relu(net)
    # frequency depthwise conv
    net = tf.keras.layers.DepthwiseConv2D(kernel_size=(1,3), strides=1, dilation_rate=dilation, padding='same', use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    residual = net
    # frequency average pooling
    net = tf.keras.backend.mean(net, axis=2, keepdims=True)
    # temporal depthwise conv
    net = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,1), strides=1, dilation_rate=dilation, padding=paddings, use_bias=False)(net)
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.activations.swish(net)
    # 1x1 conv
    net = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=1, strides=1, padding='valid', use_bias=False)(net)
    net = tf.keras.layers.SpatialDropout2D(rate=dropout)(net)
    # residual    
    net = net + residual
    net = tf.keras.activations.relu(net)

    # normal blocks:
    for _ in range(n):
        # frequency depthwise conv
        identity = net
        net = tf.keras.layers.DepthwiseConv2D(kernel_size=(1,3), strides=1, dilation_rate=dilation, padding=paddings, use_bias=False)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        residual = net
        # frequency average pooling
        net = tf.keras.backend.mean(net, axis=2, keepdims=True)
        # temporal depthwise conv
        net = tf.keras.layers.DepthwiseConv2D(kernel_size=(3,1), strides=1, dilation_rate=dilation, padding=paddings, use_bias=False)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.activations.swish(net)
        # 1x1 conv
        net = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=1, strides=1, padding=paddings, use_bias=False)(net)
        net = tf.keras.layers.SpatialDropout2D(rate=dropout)(net)
        # residual
        net = net + identity + residual
        net = tf.keras.activations.relu(net)

    if pool > 1:
      if max_pool==True:
        net = tf.keras.layers.MaxPooling2D(pool_size=(pool,1), strides=(pool,1))(net)
      else:
        net = tf.keras.layers.AveragePooling2D(pool_size=(pool,1), strides=(pool,1))(net)

  net = tf.keras.layers.DepthwiseConv2D(kernel_size=5, padding=paddings)(net)

  # average out freq dim 
  net = tf.keras.backend.mean(net, axis=2, keepdims=True)
  
  net = tf.keras.layers.Conv2D(filters=last_filters, kernel_size=1, use_bias=False)(net)

  # average out time dim
  net = tf.keras.layers.GlobalAveragePooling2D(keepdims=True)(net)
  #net = tf.keras.backend.expand_dims(net, axis=1)
  #net = tf.keras.backend.expand_dims(net, axis=1)

  net = tf.keras.layers.Conv2D(
      filters=label_count, kernel_size=1, use_bias=False)(
          net)
  # 1 and 2 dims are equal to 1
  net = tf.squeeze(net, [1, 2])

  if return_softmax==True:
    net = tf.keras.layers.Activation('softmax')(net)

  return tf.keras.Model(input_audio, net)

