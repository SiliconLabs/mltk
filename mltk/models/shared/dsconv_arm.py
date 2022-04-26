

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, AveragePooling2D
from tensorflow.keras.regularizers import l2

#define model
def DepthwiseSeparableConv2D_ARM(
    input_shape:tuple=(50,10,1), 
    num_classes:int=12,
    filters=64, 
    regularizer=l2(1e-4)
) -> Model:
    """ARM DepthwiseConv2D for Keyword Spotting

    .. seealso::
       * https://arxiv.org/pdf/1711.07128.pdf
       * https://github.com/ARM-software/ML-KWS-for-MCU
       * https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/KWS10_ARM_DSConv/dsconv_arm_eembc.py

    """

    # Model layers
    # Input pure conv2d
    inputs = Input(shape=input_shape)
    x = Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)

    # First layer of separable depthwise conv2d
    # Separable consists of depthwise conv2d followed by conv2d with 1x1 kernels
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Third layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fourth layer of separable depthwise conv2d
    x = DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Reduce size and apply final softmax
    x = Dropout(rate=0.4)(x)

    x = AveragePooling2D(pool_size=(25,5))(x)

    x = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model