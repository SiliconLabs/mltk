"""TENet - Temporal efficient neural network
***********************************************

This is a keyword spotting architecture with temporal and depthwise convolutions.

   Li, Wei, Qin: Small-Footprint Keyword Spotting with Multi-Scale Temporal Convolution, `https://arxiv.org/pdf/2010.09960.pdf`_
"""
from typing import Union, List, Tuple
import numpy as np
import tensorflow as tf


def TENet(
    input_shape:Union[List[int],Tuple[int]],
    classes: int,
    channels: int = 32,
    blocks: int = 3,
    block_depth: int = 4,
    scales: List[int] = [9],
    channel_increase: float = 0.0,
    *args,
    **kwargs,
) -> tf.keras.Model:
    """Temporal efficient neural network (TENet)
    
    A network for processing spectrogram data using temporal and depthwise convolutions.
    The network treats the [T, F] spectrogram as a timeseries shaped [T, 1, F].
    
    .. note:: When building the model, make sure that the input shape is concrete,
        i.e. explicitly reshape the samples to [T, 1, F] in the preprocessing pipeline.

    .. seealso::
       * https://arxiv.org/pdf/2010.09960.pdf

    Args:
        classes: Number of classes the network is built to categorize
        channels: Base number of channels in the network
        blocks: Number of (StridedIBB -> IBB -> ...) blocks in the networks
        block_depth: Number of IBBs inside each (StridedIBB -> IBB -> ...) block, including the strided IBB
        scales: The multitemporal convolution filter widths. Should be odd numbers >= 3.
        channel_increase: If nonzero, the network increases the channel size each time there is a strided IBB block.
                            The increase (each time) is given by `channels * channel_increase`.
"""
    count_layers = blocks * block_depth

    if isinstance(input_shape, (tuple, list)):
        input_shape = tf.TensorShape(input_shape)
    if not isinstance(input_shape, tf.TensorShape):
        raise ValueError("Invalid input_shape: Expected only one input")
    if not input_shape.is_compatible_with((None, 1, None)):
        raise ValueError(
            f"Invalid input_shape: Expected (T, 1, C) but received {input_shape}"
        )

    model_input = tf.keras.layers.Input(shape=input_shape)
    x = model_input
    x = tf.keras.layers.Conv2D(
        channels,
        (3, 1),
        padding="same",
        use_bias=True,
    )(x)


    for layer in range(count_layers):
        x = InvertedBottleneckBlock(
            x,
            channels=int(
                channels
                * (1 + channel_increase * (1 + layer // block_depth))
            ),
            stride=2 if ((layer % block_depth) == 0) else 1,
            scales=scales,
        )

    #x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.AveragePooling2D(pool_size=(x.shape[1], 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Dense(
        classes, activation=tf.keras.activations.softmax
    )(x)

    model = tf.keras.models.Model(model_input, x, name="TENet")

    return model



def TENet12(
    input_shape,
    classes: int, 
    mtconv: bool = False, 
    **kwargs
) -> tf.keras.Model:
    return TENet(
        input_shape=input_shape,
        classes=classes,
        channels=32,
        blocks=3,
        block_depth=4,
        scales=[9, 7, 5, 3] if mtconv else [9],
        **kwargs,
    )


def TENet6(
    input_shape,
    classes: int,
    mtconv: bool = False, 
    **kwargs
) -> tf.keras.Model:
    return TENet(
        input_shape=input_shape,
        classes=classes,
        channels=32,
        blocks=3,
        block_depth=2,
        scales=[9, 7, 5, 3] if mtconv else [9],
        **kwargs,
    )

def TENet12Narrow(
    input_shape,
    classes: int, 
    mtconv: bool = False, 
    *kwargs
) -> tf.keras.Model:
    return TENet(
        input_shape=input_shape,
        classes=classes,
        channels=16,
        blocks=3,
        block_depth=4,
        scales=[9, 7, 5, 3] if mtconv else [9],
        **kwargs,
    )

def TENet6Narrow(
    input_shape,
    classes: int, 
    mtconv: bool = False, 
    **kwargs
) -> tf.keras.Model:
    return TENet(
        input_shape=input_shape,
        classes=classes,
        channels=16,
        blocks=3,
        block_depth=2,
        scales=[9, 7, 5, 3] if mtconv else [9],
        **kwargs,
    )
    
def HFTENet12(
    input_shape,
    classes: int, 
    mtconv: bool = False, 
    **kwargs
) -> tf.keras.Model:
    "Custom TENet variant with channels that increase as the time axis shrinks."
    return TENet(
        input_shape=input_shape,
        classes=classes,
        channels=32,
        blocks=3,
        block_depth=4,
        scales=[9, 7, 5, 3] if mtconv else [9],
        channel_increase=0.125,
        **kwargs,
    )



class MultiScaleTemporalConvolution(tf.keras.layers.Layer):
    """Convolution which combines results form several filter widths with padding.
    Convolution is a linearly separable operation, so these different filters can be superimposed during inference.
    This makes it possible to flatten the different filters into one.
    Call `.fuse` after training to flatten the filters into one.
    """
    _layer_counter:int = 0

    def __init__(
        self,
        stride: int = 1,
        scales: Union[int, List[int], Tuple[int, ...]] = None,
        **kwargs,
    ):
        scales = scales or [3, 5, 7, 9]

        if isinstance(scales, int):
            scales = [scales]
        if not isinstance(scales, (list, tuple)):
            raise TypeError(
                f"Expected scales to be an int or a tuple/list of ints, but received {type(scales)}"
            )
        if len(scales) < 1:
            raise ValueError("Expected atleast one temporal scale, received 0")
        if any(scale < 3 for scale in scales):
            raise ValueError(f"Expected scales >= 3, received {scales}")
        if any((scale % 2) != 1 for scale in scales):
            raise ValueError(f"Expected odd scales, received {scales}")

        self.scales = list(scales)
        self.stride = stride
        self.temporal_convolutions: List[tf.keras.layers.DepthwiseConv2D] = []
        self._input_shape = None
        super().__init__(**kwargs)
    
    def get_config(self):
        config = super(MultiScaleTemporalConvolution, self).get_config()
        config.update({"stride": self.stride})
        config.update({"scales": self.scales})

        return config


    def build(self, input_shape: Union[tf.TensorShape, List[tf.TensorShape]]):
        self._input_shape = input_shape
        self.temporal_convolutions = [
            tf.keras.layers.DepthwiseConv2D(
                kernel_size=(scale, 1),
                strides=self.stride,
                padding="same",
                use_bias=False,
                name=f"mtconv{scale}-{self._layer_counter}",
                input_shape=input_shape[1:],
            )
            for scale in self.scales
        ]
        self._layer_counter += 1

        for branch in self.temporal_convolutions:
            branch.build(input_shape)
        
        return super().build(input_shape)

    def fuse(self):
        """Fuse convolutions in-place"""
        fused_convolution = self.fused()
        scale = max(self.scales)
        self.temporal_convolutions = [fused_convolution]
        self.scales = [scale]
        super().build(self._input_shape)

    def fused(self) -> tf.keras.layers.DepthwiseConv2D:
        """Returns a new depthwise conv, created by fusuing the filters of each temporal convolution in this MTConv block."""
        argmax_scale: int = tf.argmax(self.scales)
        max_scale: int = self.scales[argmax_scale]
        fused_convolution = tf.keras.layers.DepthwiseConv2D(
            kernel_size=(max_scale, 1),
            strides=self.stride,
            padding="same",
            use_bias=False,
            name=f"mtconv_superimposed-{self._layer_counter}",
        )
        self._layer_counter += 1

        fused_convolution.build(self._input_shape)
        fused_convolution.set_weights(
            self.temporal_convolutions[argmax_scale].get_weights()
        )
        for i, (scale, branch) in enumerate(
            zip(self.scales, self.temporal_convolutions)
        ):
            if i == argmax_scale:
                continue
            weights: np.ndarray = branch.get_weights()[0]
            assert weights.shape[0] == scale, f"Unexpected weight shape {weights.shape}"
            pad = (max_scale - scale) // 2
            padded_weights = np.pad(
                weights,
                pad_width=np.array([(pad, pad), (0, 0), (0, 0), (0, 0)]),
                mode="constant",
            )
            updated_weights = fused_convolution.get_weights()[0] + padded_weights
            fused_convolution.set_weights([updated_weights])
        return fused_convolution

    def compute_output_shape(self, input_shape):
        return self.temporal_convolutions[0].compute_output_shape(input_shape)

    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:
        branches = [conv(inputs) for conv in self.temporal_convolutions]
        if len(branches) == 1:
            return branches[0]  # type: ignore
        output = tf.stack(branches, axis=0)
        return tf.reduce_sum(output, axis=0)



def InvertedBottleneckBlock(
    x: tf.keras.layers.Layer,
    channels: int,
    stride: int,
    expansion_ratio: Union[float, int] = 3,
    scales: List[int] = None,
) -> tf.keras.layers.Layer:
    """Inverted bottleneck with depthwise separable temporal convolution and a residual connection."""

    input_shape = x.shape
    if not len(input_shape) == 4 and input_shape[-2] == 1:
        raise ValueError(
            f"Invalid input_shape: Exected (N, T, 1, C) but received {input_shape}"
        )
    if stride == 1 and not input_shape[-1] == channels:
        raise ValueError(
            f"Channel change is only supported for strided layers. "
            f"Expected input_shape channels ({input_shape[-1]}) "
            f"to match the bottleneck output channels ({channels}"
        )

    layer_id = globals().get('InvertedBottleneckBlock_layer_id', 0)
    globals()['InvertedBottleneckBlock_layer_id'] = layer_id + 1

    expansion_channels = int(channels * expansion_ratio)
    scales = scales or [9]

    layer_input = x
    x = tf.keras.layers.Conv2D(
        filters=expansion_channels,
        kernel_size=(1, 1),
        strides=1,
        use_bias=False,
        name=f"pointwise_expand_conv-{layer_id}",
    )(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = MultiScaleTemporalConvolution(
        stride=stride,
        scales=scales,
        name=f"mtconv-{layer_id}",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(
        filters=channels,
        kernel_size=(1, 1),
        strides=1,
        use_bias=False,
        name=f"pointwise_contract_conv-{layer_id}",
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    bottleneck_temporal_depth_separable_convolution = x

    if stride == 1:
        x = tf.keras.layers.Add()([bottleneck_temporal_depth_separable_convolution, layer_input])

    else:
        residuals = tf.keras.layers.Conv2D(
            filters=channels,
            kernel_size=(1, 1),
            strides=stride,
            padding="same",
            use_bias=False,
            name=f"strided_residual-{layer_id}",
        )(layer_input)
        residuals = tf.keras.layers.BatchNormalization()(residuals)
        residuals = tf.keras.layers.ReLU()(residuals)
        x = tf.keras.layers.Add()([bottleneck_temporal_depth_separable_convolution, residuals])

    x = tf.keras.layers.ReLU()(x)

    return x 
