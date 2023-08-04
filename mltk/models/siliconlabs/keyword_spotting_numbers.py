"""keyword_spotting_numbers
*******************************

This model is a CNN+LSTM classifier to detect the keywords:

- zero
- one
- two
- three
- four
- five
- six
- seven
- eight
- nine

This model specification script is designed to work with the
`Quantized LSTM <https://siliconlabs.github.io/mltk/mltk/tutorials/quantized_lstm.html>`_ tutorial.

- Source code: `keyword_spotting_numbers.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_numbers.py>`_
- Pre-trained model: `keyword_spotting_numbers.mltk.zip <https://github.com/SiliconLabs/mltk/raw/master/mltk/models/siliconlabs/keyword_spotting_numbers.mltk.zip>`_


Dataset
---------

This model was trained using several different datasets:

- `mltk.datasets.audio.ten_digits <https://siliconlabs.github.io/mltk/docs/python_api/datasets/audio/ten_digits.html>`_ - Synthetically generated keywords: zero, one, two, three, four, five, six, seven, eight, nine
- `mltk.datasets.audio.speech_commands_v2 <https://siliconlabs.github.io/mltk/docs/python_api/datasets/audio/speech_commands_v2.html>`_ - Human generated keywords: zero, one, two, three, four, five, six, seven, eight, nine
- `mltk.datasets.audio.mlcommons.ml_commons_keyword <https://siliconlabs.github.io/mltk/docs/python_api/datasets/audio/ml_commons/keywords.html>`_ - Large collection of keywords, random subset used for *unknown* class
- `mltk.datasets.audio.background_noise.esc50 <https://siliconlabs.github.io/mltk/docs/python_api/datasets/audio/background_noise/esc50.html>`_ - Collection of various noises, random subset used for *unknown* class
- `mltk.datasets.audio.background_noise.ambient <https://siliconlabs.github.io/mltk/docs/python_api/datasets/audio/background_noise/ambient.html>`_ - Collection of various background noises, mixed into other samples for augmentation
- `mltk.datasets.audio.background_noise.brd2601 <https://siliconlabs.github.io/mltk/docs/python_api/datasets/audio/background_noise/brd2601.html>`_ - "Silence" recorded by BRD2601 microphone, mixed into other samples to make them "sound" like they came from the BRD2601's microphone
- `mltk.datasets.audio.mit_ir_survey <https://siliconlabs.github.io/mltk/docs/python_api/datasets/audio/mit_ir_survey.html>`_ Impulse responses that are randomly convolved with the samples. This makes the samples sound if they were recorded in different environments.


.. hint::

   Uncomment the line:

   .. highlight:: python
   .. code-block:: python

      #data_dump_dir = my_model.create_log_dir('dataset_dump')

   To dump the augmented audio samples and corresponding spectrograms.
   This is useful to see how the augmentations affect the samples during training.
   WARNING: This will generate A LOT of file dumps, so be sure to disable during actual model training.


Dataset Summary
^^^^^^^^^^^^^^^^^

::

    Dataset subset: training, found 103597 samples:
         zero: 9387
          one: 9250
          two: 9245
        three: 9116
         four: 9135
         five: 9388
          six: 9229
        seven: 9342
        eight: 9166
         nine: 9289
    _unknown_: 11050
    Dataset subset: validation, found 18231 samples:
         zero: 1657
          one: 1632
          two: 1627
        three: 1603
         four: 1585
         five: 1656
          six: 1623
        seven: 1648
        eight: 1613
         nine: 1637
    _unknown_: 1950

    Class weights:
        zero = 1.00
        one = 1.02
        two = 1.02
        three = 1.03
        four = 1.03
        five = 1.00
        six = 1.02
        seven = 1.01
        eight = 1.03
        nine = 1.01
    _unknown_ = 0.85


Preprocessing
^^^^^^^^^^^^^^

The audio samples are converted to a spectrogram using the :py:class:`mltk.core.preprocess.audio.audio_feature_generator.AudioFeatureGenerator`.
The following setting are used:

- sample_rate: 16kHz
- sample_length: 1s
- window size: 30ms
- window step: 10ms
- n_channels: 104
- upper_band_limit: 7500.0
- lower_band_limit:125.0
- noise_reduction_enable: True
- noise_reduction_min_signal_remaining: 0.40
- dc_notch_filter_enable: True
- dc_notch_filter_coefficient: 0.95
- quantize_dynamic_scale_enable: False

Additionally, the uint16 spectrogram is normalized: ``spectrogram_float32 = (spectrogram - mean(spectrogram)) / std(spectrogram)``


Model Architecture
--------------------

The model is based on the `Temporal efficient neural network (TENet) <https://arxiv.org/pdf/2010.09960.pdf>`_ model architecture.

    A network for processing spectrogram data using temporal and depthwise convolutions. The network treats the [T, F] spectrogram as a timeseries shaped [T, 1, F].


More details at `mltk.models.shared.tenet.TENet <https://siliconlabs.github.io/mltk/docs/python_api/models/common_models.html#tenet>`_

Typically, the TENet model uses the AveragePool2D layer to average the frequency features across the time steps, e.g.:
``<time steps> x <frequency features> -> AveragePool2D -> 1 x frequency features>``

In this model, we replace the AveragePool2D with an LSTM, e.g.:
``<time steps> x <frequency features> -> LSTM -> 1 x frequency features>``

The idea is that rather than using simple averaging, we use an LSTM to analyze the inherent time dependencies of the frequency features across the time steps.

Note that we must also use `LayerNormalization <https://keras.io/api/layers/normalization_layers/layer_normalization>`_ at the input and output of the LSTM so that the model is properly quantized.


Overview Diagram
^^^^^^^^^^^^^^^^^

An overview of the model is illustrated as follows:

.. raw:: html

    <a href="../../../../img/keyword_spotting_numbers_model_arch.svg" target="_blank">
        <img src="../../../../img/keyword_spotting_numbers_model_arch.svg" />
    </a>


Model Summary
--------------

.. code-block:: shell

    mltk summarize keyword_spotting_numbers --tflite

    +-------+------------------------------+-------------------+-----------------+------------------------------------------------------+
    | Index | OpCode                       | Input(s)          | Output(s)       | Config                                               |
    +-------+------------------------------+-------------------+-----------------+------------------------------------------------------+
    | 0     | quantize                     | 98x1x40 (float32) | 98x1x40 (int8)  | Type=none                                            |
    | 1     | conv_2d                      | 98x1x40 (int8)    | 98x1x40 (int8)  | Padding:Same stride:1x1 activation:None              |
    |       |                              | 3x1x40 (int8)     |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 2     | conv_2d                      | 98x1x40 (int8)    | 98x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 3     | depthwise_conv_2d            | 98x1x120 (int8)   | 49x1x120 (int8) | Multiplier:1 padding:Same stride:2x2 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 4     | conv_2d                      | 49x1x120 (int8)   | 49x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 5     | conv_2d                      | 98x1x40 (int8)    | 49x1x40 (int8)  | Padding:Same stride:2x2 activation:Relu              |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 6     | add                          | 49x1x40 (int8)    | 49x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 49x1x40 (int8)    |                 |                                                      |
    | 7     | conv_2d                      | 49x1x40 (int8)    | 49x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 8     | depthwise_conv_2d            | 49x1x120 (int8)   | 49x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 9     | conv_2d                      | 49x1x120 (int8)   | 49x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 10    | add                          | 49x1x40 (int8)    | 49x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 49x1x40 (int8)    |                 |                                                      |
    | 11    | conv_2d                      | 49x1x40 (int8)    | 49x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 12    | depthwise_conv_2d            | 49x1x120 (int8)   | 49x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 13    | conv_2d                      | 49x1x120 (int8)   | 49x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 14    | add                          | 49x1x40 (int8)    | 49x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 49x1x40 (int8)    |                 |                                                      |
    | 15    | conv_2d                      | 49x1x40 (int8)    | 49x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 16    | depthwise_conv_2d            | 49x1x120 (int8)   | 49x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 17    | conv_2d                      | 49x1x120 (int8)   | 49x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 18    | add                          | 49x1x40 (int8)    | 49x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 49x1x40 (int8)    |                 |                                                      |
    | 19    | conv_2d                      | 49x1x40 (int8)    | 49x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 20    | depthwise_conv_2d            | 49x1x120 (int8)   | 25x1x120 (int8) | Multiplier:1 padding:Same stride:2x2 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 21    | conv_2d                      | 25x1x120 (int8)   | 25x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 22    | conv_2d                      | 49x1x40 (int8)    | 25x1x40 (int8)  | Padding:Same stride:2x2 activation:Relu              |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 23    | add                          | 25x1x40 (int8)    | 25x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 25x1x40 (int8)    |                 |                                                      |
    | 24    | conv_2d                      | 25x1x40 (int8)    | 25x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 25    | depthwise_conv_2d            | 25x1x120 (int8)   | 25x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 26    | conv_2d                      | 25x1x120 (int8)   | 25x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 27    | add                          | 25x1x40 (int8)    | 25x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 25x1x40 (int8)    |                 |                                                      |
    | 28    | conv_2d                      | 25x1x40 (int8)    | 25x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 29    | depthwise_conv_2d            | 25x1x120 (int8)   | 25x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 30    | conv_2d                      | 25x1x120 (int8)   | 25x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 31    | add                          | 25x1x40 (int8)    | 25x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 25x1x40 (int8)    |                 |                                                      |
    | 32    | conv_2d                      | 25x1x40 (int8)    | 25x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 33    | depthwise_conv_2d            | 25x1x120 (int8)   | 25x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 34    | conv_2d                      | 25x1x120 (int8)   | 25x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 35    | add                          | 25x1x40 (int8)    | 25x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 25x1x40 (int8)    |                 |                                                      |
    | 36    | conv_2d                      | 25x1x40 (int8)    | 25x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 37    | depthwise_conv_2d            | 25x1x120 (int8)   | 13x1x120 (int8) | Multiplier:1 padding:Same stride:2x2 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 38    | conv_2d                      | 13x1x120 (int8)   | 13x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 39    | conv_2d                      | 25x1x40 (int8)    | 13x1x40 (int8)  | Padding:Same stride:2x2 activation:Relu              |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 40    | add                          | 13x1x40 (int8)    | 13x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 13x1x40 (int8)    |                 |                                                      |
    | 41    | conv_2d                      | 13x1x40 (int8)    | 13x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 42    | depthwise_conv_2d            | 13x1x120 (int8)   | 13x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 43    | conv_2d                      | 13x1x120 (int8)   | 13x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 44    | add                          | 13x1x40 (int8)    | 13x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 13x1x40 (int8)    |                 |                                                      |
    | 45    | conv_2d                      | 13x1x40 (int8)    | 13x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 46    | depthwise_conv_2d            | 13x1x120 (int8)   | 13x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 47    | conv_2d                      | 13x1x120 (int8)   | 13x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 48    | add                          | 13x1x40 (int8)    | 13x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 13x1x40 (int8)    |                 |                                                      |
    | 49    | conv_2d                      | 13x1x40 (int8)    | 13x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 50    | depthwise_conv_2d            | 13x1x120 (int8)   | 13x1x120 (int8) | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 51    | conv_2d                      | 13x1x120 (int8)   | 13x1x40 (int8)  | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 52    | add                          | 13x1x40 (int8)    | 13x1x40 (int8)  | Activation:Relu                                      |
    |       |                              | 13x1x40 (int8)    |                 |                                                      |
    | 53    | conv_2d                      | 13x1x40 (int8)    | 13x1x120 (int8) | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 54    | depthwise_conv_2d            | 13x1x120 (int8)   | 7x1x120 (int8)  | Multiplier:1 padding:Same stride:2x2 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 55    | conv_2d                      | 7x1x120 (int8)    | 7x1x40 (int8)   | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 56    | conv_2d                      | 13x1x40 (int8)    | 7x1x40 (int8)   | Padding:Same stride:2x2 activation:Relu              |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 57    | add                          | 7x1x40 (int8)     | 7x1x40 (int8)   | Activation:Relu                                      |
    |       |                              | 7x1x40 (int8)     |                 |                                                      |
    | 58    | conv_2d                      | 7x1x40 (int8)     | 7x1x120 (int8)  | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 59    | depthwise_conv_2d            | 7x1x120 (int8)    | 7x1x120 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 60    | conv_2d                      | 7x1x120 (int8)    | 7x1x40 (int8)   | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 61    | add                          | 7x1x40 (int8)     | 7x1x40 (int8)   | Activation:Relu                                      |
    |       |                              | 7x1x40 (int8)     |                 |                                                      |
    | 62    | conv_2d                      | 7x1x40 (int8)     | 7x1x120 (int8)  | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 63    | depthwise_conv_2d            | 7x1x120 (int8)    | 7x1x120 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 64    | conv_2d                      | 7x1x120 (int8)    | 7x1x40 (int8)   | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 65    | add                          | 7x1x40 (int8)     | 7x1x40 (int8)   | Activation:Relu                                      |
    |       |                              | 7x1x40 (int8)     |                 |                                                      |
    | 66    | conv_2d                      | 7x1x40 (int8)     | 7x1x120 (int8)  | Padding:Valid stride:1x1 activation:Relu             |
    |       |                              | 1x1x40 (int8)     |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 67    | depthwise_conv_2d            | 7x1x120 (int8)    | 7x1x120 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    |       |                              | 9x1x120 (int8)    |                 |                                                      |
    |       |                              | 120 (int32)       |                 |                                                      |
    | 68    | conv_2d                      | 7x1x120 (int8)    | 7x1x40 (int8)   | Padding:Valid stride:1x1 activation:None             |
    |       |                              | 1x1x120 (int8)    |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    | 69    | add                          | 7x1x40 (int8)     | 7x1x40 (int8)   | Activation:Relu                                      |
    |       |                              | 7x1x40 (int8)     |                 |                                                      |
    | 70    | reshape                      | 7x1x40 (int8)     | 7x40x1 (int8)   | Type=none                                            |
    |       |                              | 4 (int32)         |                 |                                                      |
    | 71    | transpose                    | 7x40x1 (int8)     | 40x1x7 (int8)   | Type=none                                            |
    |       |                              | 4 (int32)         |                 |                                                      |
    | 72    | mean                         | 40x1x7 (int8)     | 7 (int8)        | Type=reduceroptions                                  |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 73    | squared_difference           | 40x1x7 (int8)     | 40x1x7 (int8)   | Type=none                                            |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 74    | mean                         | 40x1x7 (int8)     | 7 (int8)        | Type=reduceroptions                                  |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 75    | add                          | 7 (int8)          | 7 (int8)        | Activation:None                                      |
    |       |                              |  (int8)           |                 |                                                      |
    | 76    | rsqrt                        | 7 (int8)          | 7 (int8)        | Type=none                                            |
    | 77    | mul                          | 40x1x7 (int8)     | 40x1x7 (int8)   | Activation:None                                      |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 78    | mul                          | 7 (int8)          | 7 (int8)        | Activation:None                                      |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 79    | sub                          | 7 (int8)          | 7 (int8)        | Type=suboptions                                      |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 80    | add                          | 40x1x7 (int8)     | 40x1x7 (int8)   | Activation:None                                      |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 81    | transpose                    | 40x1x7 (int8)     | 7x40x1 (int8)   | Type=none                                            |
    |       |                              | 4 (int32)         |                 |                                                      |
    | 82    | reshape                      | 7x40x1 (int8)     | 7x40 (int8)     | Type=none                                            |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 83    | mul                          | 7x40 (int8)       | 7x40 (int8)     | Activation:None                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    | 84    | add                          | 7x40 (int8)       | 7x40 (int8)     | Activation:None                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    | 85    | unidirectional_sequence_lstm | 7x40 (int8)       | 7x40 (int8)     | Time major:False, Activation:Tanh, Cell clip:10.0    |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    |       |                              | 40 (int32)        |                 |                                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 40 (int16)        |                 |                                                      |
    | 86    | reshape                      | 7x40 (int8)       | 7x40x1 (int8)   | Type=none                                            |
    |       |                              | 4 (int32)         |                 |                                                      |
    | 87    | transpose                    | 7x40x1 (int8)     | 40x1x7 (int8)   | Type=none                                            |
    |       |                              | 4 (int32)         |                 |                                                      |
    | 88    | mean                         | 40x1x7 (int8)     | 7 (int8)        | Type=reduceroptions                                  |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 89    | squared_difference           | 40x1x7 (int8)     | 40x1x7 (int8)   | Type=none                                            |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 90    | mean                         | 40x1x7 (int8)     | 7 (int8)        | Type=reduceroptions                                  |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 91    | add                          | 7 (int8)          | 7 (int8)        | Activation:None                                      |
    |       |                              |  (int8)           |                 |                                                      |
    | 92    | rsqrt                        | 7 (int8)          | 7 (int8)        | Type=none                                            |
    | 93    | mul                          | 40x1x7 (int8)     | 40x1x7 (int8)   | Activation:None                                      |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 94    | mul                          | 7 (int8)          | 7 (int8)        | Activation:None                                      |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 95    | sub                          | 7 (int8)          | 7 (int8)        | Type=suboptions                                      |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 96    | add                          | 40x1x7 (int8)     | 40x1x7 (int8)   | Activation:None                                      |
    |       |                              | 7 (int8)          |                 |                                                      |
    | 97    | transpose                    | 40x1x7 (int8)     | 7x40x1 (int8)   | Type=none                                            |
    |       |                              | 4 (int32)         |                 |                                                      |
    | 98    | reshape                      | 7x40x1 (int8)     | 7x40 (int8)     | Type=none                                            |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 99    | mul                          | 7x40 (int8)       | 7x40 (int8)     | Activation:None                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    | 100   | add                          | 7x40 (int8)       | 7x40 (int8)     | Activation:None                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    | 101   | strided_slice                | 7x40 (int8)       | 40 (int8)       | Type=stridedsliceoptions                             |
    |       |                              | 3 (int32)         |                 |                                                      |
    |       |                              | 3 (int32)         |                 |                                                      |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 102   | fully_connected              | 40 (int8)         | 11 (int8)       | Activation:None                                      |
    |       |                              | 40 (int8)         |                 |                                                      |
    |       |                              | 11 (int32)        |                 |                                                      |
    | 103   | reshape                      | 11 (int8)         | 11x1x1 (int8)   | Type=none                                            |
    |       |                              | 4 (int32)         |                 |                                                      |
    | 104   | mean                         | 11x1x1 (int8)     | 1 (int8)        | Type=reduceroptions                                  |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 105   | squared_difference           | 11x1x1 (int8)     | 11x1x1 (int8)   | Type=none                                            |
    |       |                              | 1 (int8)          |                 |                                                      |
    | 106   | mean                         | 11x1x1 (int8)     | 1 (int8)        | Type=reduceroptions                                  |
    |       |                              | 3 (int32)         |                 |                                                      |
    | 107   | add                          | 1 (int8)          | 1 (int8)        | Activation:None                                      |
    |       |                              |  (int8)           |                 |                                                      |
    | 108   | rsqrt                        | 1 (int8)          | 1 (int8)        | Type=none                                            |
    | 109   | mul                          | 11x1x1 (int8)     | 11x1x1 (int8)   | Activation:None                                      |
    |       |                              | 1 (int8)          |                 |                                                      |
    | 110   | mul                          | 1 (int8)          | 1 (int8)        | Activation:None                                      |
    |       |                              | 1 (int8)          |                 |                                                      |
    | 111   | sub                          | 1 (int8)          | 1 (int8)        | Type=suboptions                                      |
    |       |                              | 1 (int8)          |                 |                                                      |
    | 112   | add                          | 11x1x1 (int8)     | 11x1x1 (int8)   | Activation:None                                      |
    |       |                              | 1 (int8)          |                 |                                                      |
    | 113   | reshape                      | 11x1x1 (int8)     | 11 (int8)       | Type=none                                            |
    |       |                              | 2 (int32)         |                 |                                                      |
    | 114   | mul                          | 11 (int8)         | 11 (int8)       | Activation:None                                      |
    |       |                              | 11 (int8)         |                 |                                                      |
    | 115   | add                          | 11 (int8)         | 11 (int8)       | Activation:None                                      |
    |       |                              | 11 (int8)         |                 |                                                      |
    | 116   | softmax                      | 11 (int8)         | 11 (int8)       | Type=softmaxoptions                                  |
    | 117   | dequantize                   | 11 (int8)         | 11 (float32)    | Type=none                                            |
    +-------+------------------------------+-------------------+-----------------+------------------------------------------------------+
    Total MACs: 5.074 M
    Total OPs: 10.306 M
    Name: keyword_spotting_numbers
    Version: 1
    Description: Keyword spotting classifier to detect: zero, one, two, three, four, five, six, seven, eight, nine
    Classes: zero, one, two, three, four, five, six, seven, eight, nine, _unknown_
    Runtime memory size (RAM): 80.500 k
    hash: 4b22adb625a3300fdcf06fa61105782f
    date: 2023-08-03T01:11:43.378Z
    fe.sample_rate_hz: 16000
    fe.fft_length: 512
    fe.sample_length_ms: 1000
    fe.window_size_ms: 30
    fe.window_step_ms: 10
    fe.filterbank_n_channels: 40
    fe.filterbank_upper_band_limit: 7500.0
    fe.filterbank_lower_band_limit: 125.0
    fe.noise_reduction_enable: True
    fe.noise_reduction_smoothing_bits: 10
    fe.noise_reduction_even_smoothing: 0.02500000037252903
    fe.noise_reduction_odd_smoothing: 0.05999999865889549
    fe.noise_reduction_min_signal_remaining: 0.4000000059604645
    fe.pcan_enable: False
    fe.pcan_strength: 0.949999988079071
    fe.pcan_offset: 80.0
    fe.pcan_gain_bits: 21
    fe.log_scale_enable: True
    fe.log_scale_shift: 6
    fe.activity_detection_enable: False
    fe.activity_detection_alpha_a: 0.5
    fe.activity_detection_alpha_b: 0.800000011920929
    fe.activity_detection_arm_threshold: 0.75
    fe.activity_detection_trip_threshold: 0.800000011920929
    fe.dc_notch_filter_enable: True
    fe.dc_notch_filter_coefficient: 0.949999988079071
    fe.quantize_dynamic_scale_enable: False
    fe.quantize_dynamic_scale_range_db: 40.0
    samplewise_norm.mean_and_std: True
    average_window_duration_ms: 450
    detection_threshold: 242
    suppression_ms: 700
    minimum_count: 2
    volume_gain: 0.0
    latency_ms: 10
    verbose_model_output_logs: False
    .tflite file size: 380.6kB


Model Profiling Report
-----------------------

.. code-block:: shell

    Profiling Summary
    Name: keyword_spotting_numbers
    Accelerator: MVP
    Input Shape: 1x98x1x40
    Input Data Type: float32
    Output Shape: 1x11
    Output Data Type: float32
    Flash, Model File Size (bytes): 379.2k
    RAM, Runtime Memory Size (bytes): 65.6k
    Operation Count: 10.5M
    Multiply-Accumulate Count: 5.1M
    Layer Count: 118
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 4.5M
    CPU Cycle Count: 3.2M
    CPU Utilization (%): 44.4
    Clock Rate (hz): 78.0M
    Time (s): 92.6m
    Ops/s: 113.5M
    MACs/s: 54.8M
    Inference/s: 10.8

    Model Layers
    +-------+------------------------------+--------+--------+------------+------------+----------+------------------------------------------------------------------------------+--------------+------------------------------------------------------+
    | Index | OpCode                       | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape                                                                  | Output Shape | Options                                              |
    +-------+------------------------------+--------+--------+------------+------------+----------+------------------------------------------------------------------------------+--------------+------------------------------------------------------+
    | 0     | quantize                     | 15.7k  | 0      | 0          | 142.2k     | 1.8m     | 1x98x1x40                                                                    | 1x98x1x40    | Type=none                                            |
    | 1     | conv_2d                      | 944.7k | 470.4k | 364.9k     | 10.8k      | 4.7m     | 1x98x1x40,40x3x1x40,40                                                       | 1x98x1x40    | Padding:Same stride:1x1 activation:None              |
    | 2     | conv_2d                      | 976.1k | 470.4k | 390.6k     | 5.1k       | 5.0m     | 1x98x1x40,120x1x1x40,120                                                     | 1x98x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 3     | depthwise_conv_2d            | 123.5k | 52.9k  | 94.4k      | 75.6k      | 1.6m     | 1x98x1x120,1x9x1x120,120                                                     | 1x49x1x120   | Multiplier:1 padding:Same stride:2x2 activation:Relu |
    | 4     | conv_2d                      | 472.4k | 235.2k | 186.5k     | 5.4k       | 2.4m     | 1x49x1x120,40x1x1x120,40                                                     | 1x49x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 5     | conv_2d                      | 162.7k | 78.4k  | 66.7k      | 5.0k       | 870.0u   | 1x98x1x40,40x1x1x40,40                                                       | 1x49x1x40    | Padding:Same stride:2x2 activation:Relu              |
    | 6     | add                          | 2.0k   | 0      | 6.9k       | 2.7k       | 120.0u   | 1x49x1x40,1x49x1x40                                                          | 1x49x1x40    | Activation:Relu                                      |
    | 7     | conv_2d                      | 488.0k | 235.2k | 195.2k     | 5.1k       | 2.5m     | 1x49x1x40,120x1x1x40,120                                                     | 1x49x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 8     | depthwise_conv_2d            | 123.5k | 52.9k  | 92.6k      | 75.5k      | 1.6m     | 1x49x1x120,1x9x1x120,120                                                     | 1x49x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 9     | conv_2d                      | 472.4k | 235.2k | 186.5k     | 5.4k       | 2.4m     | 1x49x1x120,40x1x1x120,40                                                     | 1x49x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 10    | add                          | 2.0k   | 0      | 6.9k       | 2.7k       | 120.0u   | 1x49x1x40,1x49x1x40                                                          | 1x49x1x40    | Activation:Relu                                      |
    | 11    | conv_2d                      | 488.0k | 235.2k | 195.2k     | 5.1k       | 2.5m     | 1x49x1x40,120x1x1x40,120                                                     | 1x49x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 12    | depthwise_conv_2d            | 123.5k | 52.9k  | 92.6k      | 75.5k      | 1.6m     | 1x49x1x120,1x9x1x120,120                                                     | 1x49x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 13    | conv_2d                      | 472.4k | 235.2k | 186.5k     | 5.4k       | 2.4m     | 1x49x1x120,40x1x1x120,40                                                     | 1x49x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 14    | add                          | 2.0k   | 0      | 6.9k       | 2.7k       | 120.0u   | 1x49x1x40,1x49x1x40                                                          | 1x49x1x40    | Activation:Relu                                      |
    | 15    | conv_2d                      | 488.0k | 235.2k | 195.2k     | 5.1k       | 2.5m     | 1x49x1x40,120x1x1x40,120                                                     | 1x49x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 16    | depthwise_conv_2d            | 123.5k | 52.9k  | 92.6k      | 75.5k      | 1.6m     | 1x49x1x120,1x9x1x120,120                                                     | 1x49x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 17    | conv_2d                      | 472.4k | 235.2k | 186.5k     | 5.4k       | 2.4m     | 1x49x1x120,40x1x1x120,40                                                     | 1x49x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 18    | add                          | 2.0k   | 0      | 6.9k       | 2.7k       | 120.0u   | 1x49x1x40,1x49x1x40                                                          | 1x49x1x40    | Activation:Relu                                      |
    | 19    | conv_2d                      | 488.0k | 235.2k | 195.7k     | 5.1k       | 2.5m     | 1x49x1x40,120x1x1x40,120                                                     | 1x49x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 20    | depthwise_conv_2d            | 63.0k  | 27.0k  | 46.9k      | 39.1k      | 840.0u   | 1x49x1x120,1x9x1x120,120                                                     | 1x25x1x120   | Multiplier:1 padding:Same stride:2x2 activation:Relu |
    | 21    | conv_2d                      | 241.0k | 120.0k | 95.3k      | 5.4k       | 1.2m     | 1x25x1x120,40x1x1x120,40                                                     | 1x25x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 22    | conv_2d                      | 83.0k  | 40.0k  | 34.5k      | 5.0k       | 480.0u   | 1x49x1x40,40x1x1x40,40                                                       | 1x25x1x40    | Padding:Same stride:2x2 activation:Relu              |
    | 23    | add                          | 1.0k   | 0      | 3.5k       | 2.7k       | 90.0u    | 1x25x1x40,1x25x1x40                                                          | 1x25x1x40    | Activation:Relu                                      |
    | 24    | conv_2d                      | 249.0k | 120.0k | 99.9k      | 5.1k       | 1.3m     | 1x25x1x40,120x1x1x40,120                                                     | 1x25x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 25    | depthwise_conv_2d            | 63.0k  | 27.0k  | 45.5k      | 39.2k      | 810.0u   | 1x25x1x120,1x9x1x120,120                                                     | 1x25x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 26    | conv_2d                      | 241.0k | 120.0k | 95.3k      | 5.4k       | 1.2m     | 1x25x1x120,40x1x1x120,40                                                     | 1x25x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 27    | add                          | 1.0k   | 0      | 3.5k       | 2.7k       | 90.0u    | 1x25x1x40,1x25x1x40                                                          | 1x25x1x40    | Activation:Relu                                      |
    | 28    | conv_2d                      | 249.0k | 120.0k | 99.9k      | 5.1k       | 1.3m     | 1x25x1x40,120x1x1x40,120                                                     | 1x25x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 29    | depthwise_conv_2d            | 63.0k  | 27.0k  | 45.5k      | 39.2k      | 810.0u   | 1x25x1x120,1x9x1x120,120                                                     | 1x25x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 30    | conv_2d                      | 241.0k | 120.0k | 95.3k      | 5.4k       | 1.2m     | 1x25x1x120,40x1x1x120,40                                                     | 1x25x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 31    | add                          | 1.0k   | 0      | 3.5k       | 2.7k       | 90.0u    | 1x25x1x40,1x25x1x40                                                          | 1x25x1x40    | Activation:Relu                                      |
    | 32    | conv_2d                      | 249.0k | 120.0k | 99.9k      | 5.1k       | 1.3m     | 1x25x1x40,120x1x1x40,120                                                     | 1x25x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 33    | depthwise_conv_2d            | 63.0k  | 27.0k  | 45.5k      | 39.2k      | 780.0u   | 1x25x1x120,1x9x1x120,120                                                     | 1x25x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 34    | conv_2d                      | 241.0k | 120.0k | 95.3k      | 5.4k       | 1.2m     | 1x25x1x120,40x1x1x120,40                                                     | 1x25x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 35    | add                          | 1.0k   | 0      | 3.5k       | 2.7k       | 90.0u    | 1x25x1x40,1x25x1x40                                                          | 1x25x1x40    | Activation:Relu                                      |
    | 36    | conv_2d                      | 249.0k | 120.0k | 99.9k      | 5.1k       | 1.3m     | 1x25x1x40,120x1x1x40,120                                                     | 1x25x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 37    | depthwise_conv_2d            | 32.8k  | 14.0k  | 23.4k      | 21.0k      | 420.0u   | 1x25x1x120,1x9x1x120,120                                                     | 1x13x1x120   | Multiplier:1 padding:Same stride:2x2 activation:Relu |
    | 38    | conv_2d                      | 125.3k | 62.4k  | 49.6k      | 5.3k       | 660.0u   | 1x13x1x120,40x1x1x120,40                                                     | 1x13x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 39    | conv_2d                      | 43.2k  | 20.8k  | 17.9k      | 5.0k       | 270.0u   | 1x25x1x40,40x1x1x40,40                                                       | 1x13x1x40    | Padding:Same stride:2x2 activation:Relu              |
    | 40    | add                          | 520.0  | 0      | 1.8k       | 2.7k       | 60.0u    | 1x13x1x40,1x13x1x40                                                          | 1x13x1x40    | Activation:Relu                                      |
    | 41    | conv_2d                      | 129.5k | 62.4k  | 52.0k      | 5.1k       | 720.0u   | 1x13x1x40,120x1x1x40,120                                                     | 1x13x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 42    | depthwise_conv_2d            | 32.8k  | 14.0k  | 21.9k      | 21.0k      | 390.0u   | 1x13x1x120,1x9x1x120,120                                                     | 1x13x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 43    | conv_2d                      | 125.3k | 62.4k  | 49.6k      | 5.3k       | 660.0u   | 1x13x1x120,40x1x1x120,40                                                     | 1x13x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 44    | add                          | 520.0  | 0      | 1.8k       | 2.7k       | 60.0u    | 1x13x1x40,1x13x1x40                                                          | 1x13x1x40    | Activation:Relu                                      |
    | 45    | conv_2d                      | 129.5k | 62.4k  | 52.0k      | 5.1k       | 720.0u   | 1x13x1x40,120x1x1x40,120                                                     | 1x13x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 46    | depthwise_conv_2d            | 32.8k  | 14.0k  | 21.9k      | 21.0k      | 420.0u   | 1x13x1x120,1x9x1x120,120                                                     | 1x13x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 47    | conv_2d                      | 125.3k | 62.4k  | 49.6k      | 5.3k       | 660.0u   | 1x13x1x120,40x1x1x120,40                                                     | 1x13x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 48    | add                          | 520.0  | 0      | 1.8k       | 2.7k       | 30.0u    | 1x13x1x40,1x13x1x40                                                          | 1x13x1x40    | Activation:Relu                                      |
    | 49    | conv_2d                      | 129.5k | 62.4k  | 52.0k      | 5.1k       | 690.0u   | 1x13x1x40,120x1x1x40,120                                                     | 1x13x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 50    | depthwise_conv_2d            | 32.8k  | 14.0k  | 21.9k      | 21.0k      | 390.0u   | 1x13x1x120,1x9x1x120,120                                                     | 1x13x1x120   | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 51    | conv_2d                      | 125.3k | 62.4k  | 49.6k      | 5.3k       | 690.0u   | 1x13x1x120,40x1x1x120,40                                                     | 1x13x1x40    | Padding:Valid stride:1x1 activation:None             |
    | 52    | add                          | 520.0  | 0      | 1.8k       | 2.7k       | 60.0u    | 1x13x1x40,1x13x1x40                                                          | 1x13x1x40    | Activation:Relu                                      |
    | 53    | conv_2d                      | 129.5k | 62.4k  | 52.0k      | 5.1k       | 720.0u   | 1x13x1x40,120x1x1x40,120                                                     | 1x13x1x120   | Padding:Valid stride:1x1 activation:Relu             |
    | 54    | depthwise_conv_2d            | 17.6k  | 7.6k   | 11.6k      | 11.9k      | 210.0u   | 1x13x1x120,1x9x1x120,120                                                     | 1x7x1x120    | Multiplier:1 padding:Same stride:2x2 activation:Relu |
    | 55    | conv_2d                      | 67.5k  | 33.6k  | 26.7k      | 5.3k       | 390.0u   | 1x7x1x120,40x1x1x120,40                                                      | 1x7x1x40     | Padding:Valid stride:1x1 activation:None             |
    | 56    | conv_2d                      | 23.2k  | 11.2k  | 9.7k       | 5.0k       | 180.0u   | 1x13x1x40,40x1x1x40,40                                                       | 1x7x1x40     | Padding:Same stride:2x2 activation:Relu              |
    | 57    | add                          | 280.0  | 0      | 992.0      | 2.7k       | 30.0u    | 1x7x1x40,1x7x1x40                                                            | 1x7x1x40     | Activation:Relu                                      |
    | 58    | conv_2d                      | 69.7k  | 33.6k  | 28.1k      | 5.1k       | 390.0u   | 1x7x1x40,120x1x1x40,120                                                      | 1x7x1x120    | Padding:Valid stride:1x1 activation:Relu             |
    | 59    | depthwise_conv_2d            | 17.6k  | 7.6k   | 10.2k      | 11.9k      | 210.0u   | 1x7x1x120,1x9x1x120,120                                                      | 1x7x1x120    | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 60    | conv_2d                      | 67.5k  | 33.6k  | 26.7k      | 5.3k       | 390.0u   | 1x7x1x120,40x1x1x120,40                                                      | 1x7x1x40     | Padding:Valid stride:1x1 activation:None             |
    | 61    | add                          | 280.0  | 0      | 992.0      | 2.7k       | 30.0u    | 1x7x1x40,1x7x1x40                                                            | 1x7x1x40     | Activation:Relu                                      |
    | 62    | conv_2d                      | 69.7k  | 33.6k  | 28.1k      | 5.1k       | 390.0u   | 1x7x1x40,120x1x1x40,120                                                      | 1x7x1x120    | Padding:Valid stride:1x1 activation:Relu             |
    | 63    | depthwise_conv_2d            | 17.6k  | 7.6k   | 10.2k      | 11.9k      | 210.0u   | 1x7x1x120,1x9x1x120,120                                                      | 1x7x1x120    | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 64    | conv_2d                      | 67.5k  | 33.6k  | 26.7k      | 5.3k       | 390.0u   | 1x7x1x120,40x1x1x120,40                                                      | 1x7x1x40     | Padding:Valid stride:1x1 activation:None             |
    | 65    | add                          | 280.0  | 0      | 992.0      | 2.7k       | 30.0u    | 1x7x1x40,1x7x1x40                                                            | 1x7x1x40     | Activation:Relu                                      |
    | 66    | conv_2d                      | 69.7k  | 33.6k  | 28.1k      | 5.1k       | 390.0u   | 1x7x1x40,120x1x1x40,120                                                      | 1x7x1x120    | Padding:Valid stride:1x1 activation:Relu             |
    | 67    | depthwise_conv_2d            | 17.6k  | 7.6k   | 10.2k      | 11.9k      | 210.0u   | 1x7x1x120,1x9x1x120,120                                                      | 1x7x1x120    | Multiplier:1 padding:Same stride:1x1 activation:Relu |
    | 68    | conv_2d                      | 67.5k  | 33.6k  | 26.7k      | 5.3k       | 390.0u   | 1x7x1x120,40x1x1x120,40                                                      | 1x7x1x40     | Padding:Valid stride:1x1 activation:None             |
    | 69    | add                          | 280.0  | 0      | 992.0      | 2.7k       | 60.0u    | 1x7x1x40,1x7x1x40                                                            | 1x7x1x40     | Activation:Relu                                      |
    | 70    | reshape                      | 0      | 0      | 0          | 1.8k       | 0        | 1x7x1x40,4                                                                   | 1x7x40x1     | Type=none                                            |
    | 71    | transpose                    | 0      | 0      | 0          | 9.2k       | 90.0u    | 1x7x40x1,4                                                                   | 1x40x1x7     | Type=none                                            |
    | 72    | mean                         | 0      | 0      | 0          | 52.7k      | 660.0u   | 1x40x1x7,3                                                                   | 7            | Type=reduceroptions                                  |
    | 73    | squared_difference           | 0      | 0      | 0          | 31.7k      | 390.0u   | 1x40x1x7,7                                                                   | 1x40x1x7     | Type=none                                            |
    | 74    | mean                         | 0      | 0      | 0          | 52.3k      | 660.0u   | 1x40x1x7,3                                                                   | 7            | Type=reduceroptions                                  |
    | 75    | add                          | 7.0    | 0      | 0          | 2.1k       | 30.0u    | 7,                                                                           | 7            | Activation:None                                      |
    | 76    | rsqrt                        | 0      | 0      | 0          | 4.8k       | 60.0u    | 7                                                                            | 7            | Type=none                                            |
    | 77    | mul                          | 280.0  | 0      | 0          | 23.3k      | 270.0u   | 1x40x1x7,7                                                                   | 1x40x1x7     | Activation:None                                      |
    | 78    | mul                          | 7.0    | 0      | 32.0       | 2.5k       | 60.0u    | 7,7                                                                          | 7            | Activation:None                                      |
    | 79    | sub                          | 0      | 0      | 0          | 1.8k       | 0        | 7,7                                                                          | 7            | Type=suboptions                                      |
    | 80    | add                          | 280.0  | 0      | 0          | 25.4k      | 300.0u   | 1x40x1x7,7                                                                   | 1x40x1x7     | Activation:None                                      |
    | 81    | transpose                    | 0      | 0      | 0          | 28.4k      | 360.0u   | 1x40x1x7,4                                                                   | 1x7x40x1     | Type=none                                            |
    | 82    | reshape                      | 0      | 0      | 0          | 1.8k       | 30.0u    | 1x7x40x1,3                                                                   | 1x7x40       | Type=none                                            |
    | 83    | mul                          | 280.0  | 0      | 0          | 23.1k      | 300.0u   | 1x7x40,40                                                                    | 1x7x40       | Activation:None                                      |
    | 84    | add                          | 280.0  | 0      | 0          | 24.4k      | 300.0u   | 1x7x40,40                                                                    | 1x7x40       | Activation:None                                      |
    | 85    | unidirectional_sequence_lstm | 0      | 0      | 0          | 1.6M       | 20.5m    | 1x7x40,40x40,40x40,40x40,40x40,40x40,40x40,40x40,40x40,40,40,40,40,1x40,1x40 | 1x7x40       | Time major:False, Activation:Tanh, Cell clip:10.0    |
    | 86    | reshape                      | 0      | 0      | 0          | 1.8k       | 30.0u    | 1x7x40,4                                                                     | 1x7x40x1     | Type=none                                            |
    | 87    | transpose                    | 0      | 0      | 0          | 9.2k       | 120.0u   | 1x7x40x1,4                                                                   | 1x40x1x7     | Type=none                                            |
    | 88    | mean                         | 0      | 0      | 0          | 52.6k      | 660.0u   | 1x40x1x7,3                                                                   | 7            | Type=reduceroptions                                  |
    | 89    | squared_difference           | 0      | 0      | 0          | 31.6k      | 390.0u   | 1x40x1x7,7                                                                   | 1x40x1x7     | Type=none                                            |
    | 90    | mean                         | 0      | 0      | 0          | 52.3k      | 660.0u   | 1x40x1x7,3                                                                   | 7            | Type=reduceroptions                                  |
    | 91    | add                          | 7.0    | 0      | 0          | 2.1k       | 0        | 7,                                                                           | 7            | Activation:None                                      |
    | 92    | rsqrt                        | 0      | 0      | 0          | 4.8k       | 60.0u    | 7                                                                            | 7            | Type=none                                            |
    | 93    | mul                          | 280.0  | 0      | 0          | 23.3k      | 270.0u   | 1x40x1x7,7                                                                   | 1x40x1x7     | Activation:None                                      |
    | 94    | mul                          | 7.0    | 0      | 32.0       | 2.6k       | 30.0u    | 7,7                                                                          | 7            | Activation:None                                      |
    | 95    | sub                          | 0      | 0      | 0          | 1.8k       | 30.0u    | 7,7                                                                          | 7            | Type=suboptions                                      |
    | 96    | add                          | 280.0  | 0      | 0          | 25.4k      | 300.0u   | 1x40x1x7,7                                                                   | 1x40x1x7     | Activation:None                                      |
    | 97    | transpose                    | 0      | 0      | 0          | 28.4k      | 330.0u   | 1x40x1x7,4                                                                   | 1x7x40x1     | Type=none                                            |
    | 98    | reshape                      | 0      | 0      | 0          | 1.8k       | 0        | 1x7x40x1,3                                                                   | 1x7x40       | Type=none                                            |
    | 99    | mul                          | 280.0  | 0      | 0          | 23.1k      | 300.0u   | 1x7x40,40                                                                    | 1x7x40       | Activation:None                                      |
    | 100   | add                          | 280.0  | 0      | 0          | 24.4k      | 330.0u   | 1x7x40,40                                                                    | 1x7x40       | Activation:None                                      |
    | 101   | strided_slice                | 0      | 0      | 0          | 2.1k       | 30.0u    | 1x7x40,3,3,3                                                                 | 1x40         | Type=stridedsliceoptions                             |
    | 102   | fully_connected              | 891.0  | 440.0  | 749.0      | 2.2k       | 30.0u    | 1x40,11x40,11                                                                | 1x11         | Activation:None                                      |
    | 103   | reshape                      | 0      | 0      | 0          | 464.0      | 0        | 1x11,4                                                                       | 1x11x1x1     | Type=none                                            |
    | 104   | mean                         | 0      | 0      | 0          | 4.1k       | 60.0u    | 1x11x1x1,3                                                                   | 1            | Type=reduceroptions                                  |
    | 105   | squared_difference           | 0      | 0      | 0          | 3.2k       | 60.0u    | 1x11x1x1,1                                                                   | 1x11x1x1     | Type=none                                            |
    | 106   | mean                         | 0      | 0      | 0          | 3.7k       | 30.0u    | 1x11x1x1,3                                                                   | 1            | Type=reduceroptions                                  |
    | 107   | add                          | 1.0    | 0      | 0          | 1.3k       | 30.0u    | 1,                                                                           | 1            | Activation:None                                      |
    | 108   | rsqrt                        | 0      | 0      | 0          | 1.2k       | 30.0u    | 1                                                                            | 1            | Type=none                                            |
    | 109   | mul                          | 11.0   | 0      | 0          | 2.9k       | 30.0u    | 1x11x1x1,1                                                                   | 1x11x1x1     | Activation:None                                      |
    | 110   | mul                          | 1.0    | 0      | 0          | 1.5k       | 30.0u    | 1,1                                                                          | 1            | Activation:None                                      |
    | 111   | sub                          | 0      | 0      | 0          | 1.2k       | 0        | 1,1                                                                          | 1            | Type=suboptions                                      |
    | 112   | add                          | 11.0   | 0      | 0          | 3.4k       | 60.0u    | 1x11x1x1,1                                                                   | 1x11x1x1     | Activation:None                                      |
    | 113   | reshape                      | 0      | 0      | 0          | 460.0      | 0        | 1x11x1x1,2                                                                   | 1x11         | Type=none                                            |
    | 114   | mul                          | 11.0   | 0      | 49.0       | 2.6k       | 30.0u    | 1x11,11                                                                      | 1x11         | Activation:None                                      |
    | 115   | add                          | 11.0   | 0      | 50.0       | 2.4k       | 30.0u    | 1x11,11                                                                      | 1x11         | Activation:None                                      |
    | 116   | softmax                      | 55.0   | 0      | 0          | 7.8k       | 90.0u    | 1x11                                                                         | 1x11         | Type=softmaxoptions                                  |
    | 117   | dequantize                   | 22.0   | 0      | 0          | 1.5k       | 30.0u    | 1x11                                                                         | 1x11         | Type=none                                            |
    +-------+------------------------------+--------+--------+------------+------------+----------+------------------------------------------------------------------------------+--------------+------------------------------------------------------+


Model Evaluation
-----------------------

.. code-block:: shell

   # Evaluate float32 (i.e. .h5) model
   mltk evaluate keyword_spotting_numbers

    Name: keyword_spotting_numbers
    Model Type: classification
    Overall accuracy: 93.840%
    Class accuracies:
    - seven = 96.723%
    - eight = 96.404%
    - nine = 94.746%
    - six = 94.701%
    - zero = 94.508%
    - three = 94.198%
    - two = 93.915%
    - one = 93.873%
    - four = 93.249%
    - five = 90.882%
    - _unknown_ = 89.846%
    Average ROC AUC: 99.241%
    Class ROC AUC:
    - seven = 99.723%
    - eight = 99.715%
    - four = 99.342%
    - one = 99.329%
    - zero = 99.294%
    - nine = 99.289%
    - three = 99.260%
    - six = 99.227%
    - two = 99.080%
    - _unknown_ = 98.756%
    - five = 98.636%


.. code-block:: shell

   # Evaluate int8 (i.e. .tflite) model
   mltk evaluate keyword_spotting_numbers --tflite

    Name: keyword_spotting_numbers
    Model Type: classification
    Overall accuracy: 90.116%
    Class accuracies:
    - seven = 94.478%
    - six = 94.023%
    - three = 92.764%
    - zero = 92.215%
    - eight = 91.135%
    - nine = 90.165%
    - two = 90.043%
    - one = 88.848%
    - four = 88.265%
    - five = 86.836%
    - _unknown_ = 83.744%
    Average ROC AUC: 98.535%
    Class ROC AUC:
    - seven = 99.258%
    - three = 98.892%
    - six = 98.784%
    - two = 98.709%
    - zero = 98.701%
    - nine = 98.615%
    - eight = 98.611%
    - four = 98.457%
    - one = 98.379%
    - five = 97.937%
    - _unknown_ = 97.545%


Model Diagram
------------------

.. code-block:: shell

   mltk view keyword_spotting_numbers --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/keyword_spotting_numbers.tflite.png" target="_blank">
            <img src="../../../../_images/models/keyword_spotting_numbers.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train keyword_spotting_numbers-test

   # Train the model
   mltk train keyword_spotting_numbers

   # Evaluate the trained model .tflite model
   mltk evaluate keyword_spotting_numbers --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile keyword_spotting_numbers --accelerator MVP --estimates

   # Profile the model on a physical development board
   mltk profile keyword_spotting_numbers  --accelerator MVP --device

   # Run the model in the audio classifier on the local PC
   mltk classify_audio keyword_spotting_numbers --verbose

   # Run the model in the audio classifier on the physical device
   mltk classify_audio keyword_spotting_numbers --device --verbose --accelerator MVP


Model Specification
---------------------

..  literalinclude:: ../../../../../../../mltk/models/siliconlabs/keyword_spotting_numbers.py
    :language: python
    :lines: 788-

"""
# pylint: disable=redefined-outer-name

# Import the Tensorflow packages
# required to build the model layout
import os
import math
from typing import Tuple, Dict, List


import numpy as np
import tensorflow as tf
import mltk.core as mltk_core

# Import the AudioFeatureGeneratorSettings which we'll configure
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.core.preprocess.utils import tf_dataset as tf_dataset_utils
from mltk.core.preprocess.utils import audio as audio_utils
from mltk.core.preprocess.utils import image as image_utils
from mltk.core.keras.callbacks import SteppedLearnRateScheduler
from mltk.utils.path import create_user_dir
from mltk.core.preprocess.utils import (split_file_list, shuffle_file_list_by_group)
from mltk.utils.python import install_pip_package
from mltk.models.shared import tenet
from mltk.datasets import audio as audio_datasets


##########################################################################################
# Instantiate the MltkModel instance
#

# @mltk_model
class MyModel(
    mltk_core.MltkModel,    # We must inherit the MltkModel class
    mltk_core.TrainMixin,   # We also inherit the TrainMixin since we want to train this model
    mltk_core.DatasetMixin, # We also need the DatasetMixin mixin to provide the relevant dataset properties
    mltk_core.EvaluateClassifierMixin,  # While not required, also inherit EvaluateClassifierMixin to help will generating evaluation stats for our classification model
):
    pass
# Instantiate our custom model object
# The rest of this script simply configures the properties
# of our custom model object
my_model = MyModel()

##########################################################################################
# General Settings

# For better tracking, the version should be incremented any time a non-trivial change is made
# NOTE: The version is optional and not used directly used by the MLTK
my_model.version = 1
# Provide a brief description about what this model models
# This description goes in the "description" field of the .tflite model file
my_model.description = 'Keyword spotting classifier to detect: zero, one, two, three, four, five, six, seven, eight, nine'


##########################################################################################
# Training Basic Settings

# This specifies the number of times we run the training.
# We just set this to a large value since we're using SteppedLearnRateScheduler
# to control when training completes
my_model.epochs = 9999
# Specify how many samples to pass through the model
# before updating the training gradients.
# Typical values are 10-64
# NOTE: Larger values require more memory and may not fit on your GPU
my_model.batch_size = 64


##########################################################################################
# Training callback Settings
#


# The MLTK enables the tf.keras.callbacks.ModelCheckpoint by default.
my_model.checkpoint['monitor'] =  'val_accuracy'


# We use a custom learn rate schedule that is defined in:
# https://github.com/google-research/google-research/tree/master/kws_streaming
my_model.train_callbacks = [
    tf.keras.callbacks.TerminateOnNaN(),
    SteppedLearnRateScheduler([
        (5000,   .001),
        (5000,   .002),
        (5000,   .003),
        (5000,   .004),
        (60000, .005),
        (20000, .002),
        (20000, .0005),
        (20000, 1e-5),
        (8000,  1e-6),
        (7000,  1e-7),
    ] )
]

##########################################################################################
# TF-Lite converter settings
#

# These are the settings used to quantize the model.
# We want all the internal ops to use int8
# while the model input/output is float32.
# (the TfliteConverter will automatically add the quantize/dequantize layers)
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# We are normalizing the input samples, so the input/output must be float32
my_model.tflite_converter['inference_input_type'] = np.float32
my_model.tflite_converter['inference_output_type'] = np.float32
# Automatically generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'
# Use 1000 samples from each class to determine the quantization ranges
my_model.tflite_converter['representative_dataset_max_samples'] = 10*1000
# Generate a quantization report to help with debugging quantization errors
my_model.tflite_converter['generate_quantization_report'] = True


##########################################################################################
# Define the model architecture
#

# This is the number of spectrogram frequency bins to use throughout the model
n_frequency_bins = 40


def my_model_builder(model: MyModel, batch_size:int=None) -> tf.keras.Model:
    """Build the Keras model

    NOTE: batch_size=None by default so that the batch can be dynamically set during model training.
          When the model is saved to .h5, batch_size=1 which is necessary when generating the quantized .tflite.
    """
    input_shape = model.input_shape
    # NOTE: The TENet model requires the input shape: <time, 1, features>
    #       while the embedded device expects: <time, features, 1>
    #       Since the <time> axis is still row-major, we can swap the <features> with 1 without issue
    time_size, feature_size, _ = input_shape
    input_shape = (time_size, 1, feature_size)

    # If specified, force the model input's batch_size to the given value
    input_layer = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

    # Create the TENet model WITHOUT the classifier head
    x = tenet.TENet12(
        input_shape=input_shape,
        classes=model.n_classes,
        channels=feature_size,
        blocks=4,
        include_head=False,
        return_model=False,
        input_layer=input_layer,
    )

    # Determine the output shape of the TENet model above
    # These are the CNN features extracted from the input spectrogram:
    # <batch_size, cnn_time_steps, 1, n_frequency_bins>
    _, cnn_time_steps, _, _ = x.shape

    # The input to the LSTM layer is:
    # <batch_size, cnn_time_steps, n_frequency_bins>
    x = tf.keras.layers.Reshape((cnn_time_steps, n_frequency_bins))(x)

    # It is critical that we normalize the LSTM input, i.e.
    # normalized_lstm_input = (cnn_features - mean(cnn_features)) / std(cnn_features)
    # This helps to ensure that the LSTM layer is properly quantized.
    x = tf.keras.layers.LayerNormalization()(x)

    # The TENet typically uses AveragePooling2D to average the frequency bins across the time steps.
    # In this model, we use an LSTM layer to generate features based on the recurrent nature of the spectrogram.
    # This analyzes the patterns of the <n_frequency_bins> frequency bins along the time axis.
    x = tf.keras.layers.LSTM(
        n_frequency_bins,      # We want 1 LSTM cell for each spectrogram frequency bin
        activation='tanh',     # Embedded only supports the tanh activation
        return_sequences=True, # This is required so that the LSTM layer is properly generated in the .tflite
                               # If this is false, the a WHILE layer is used which is not optimal for embedded
        #dropout=0.1,
        #recurrent_dropout=0.1  # The recurrent_dropout param is not supported by the embedded kernel
    )(x)

    # It is critical that we normalize the LSTM output, i.e.
    # normalized_lstm_output = (lstm_output - mean(lstm_output)) / std(lstm_output)
    # This helps to ensure that the LSTM layer is properly quantized.
    x = tf.keras.layers.LayerNormalization()(x)

    # The output of the LSTM is:
    # <batch_size, cnn_time_steps, n_frequency_bins>
    # However, only the last row of the LSTM is meaningful,
    # so we drop the rest of the rows:
    # <batch_size, last_row_lstm_features>
    x = tf.keras.layers.Lambda(lambda x: x[:, -1, :])(x)

    # Classify the results of the previous fully connected layer.
    x = tf.keras.layers.Dense(model.n_classes)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Activation('softmax')(x)

    keras_model = tf.keras.Model(input_layer, x)

    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8),
        metrics= ['accuracy']
    )

    return keras_model

my_model.build_model_function = my_model_builder
# TENet uses a custom layer, be sure to add it to the keras_custom_objects
# so that we can load the corresponding .h5 model file
my_model.keras_custom_objects['MultiScaleTemporalConvolution'] = tenet.MultiScaleTemporalConvolution



def _before_save_train_model(
        mltk_model:mltk_core.MltkModel,
        keras_model:mltk_core.KerasModel,
        keras_model_dict:dict,
        **kwargs
    ):
    """This is called just before the trained model is saved to a .h5
    This forces the batch_size=1 which is necessary when quantizing the model into a .tflite.
    """
    old_weights = keras_model.get_weights()
    new_keras_model = my_model_builder(mltk_model, batch_size=1)
    new_keras_model.set_weights(old_weights)
    keras_model_dict['value'] = new_keras_model

my_model.add_event_handler(mltk_core.MltkModelEvent.BEFORE_SAVE_TRAIN_MODEL, _before_save_train_model)


def _evaluate_startup(mltk_model:mltk_core.MltkModel, **kwargs):
    """This is called at the beginning of the model evaluation API.
    This forces the batch_size=1 which is necessary as that is how the .h5 and .tflite model files were saved.
    """
    mltk_model.batch_size = 1

my_model.add_event_handler(mltk_core.MltkModelEvent.EVALUATE_STARTUP, _evaluate_startup)




##########################################################################################
# Specify AudioFeatureGenerator Settings
# See https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
#
frontend_settings = AudioFeatureGeneratorSettings()

frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1000                       # A 1s buffer should be enough to capture the keywords
frontend_settings.window_size_ms = 30
frontend_settings.window_step_ms = 10
frontend_settings.filterbank_n_channels = n_frequency_bins      # This defines the number of frequency bins generated in the spectrogram
                                                                # and processed by the ML model.
                                                                # Typically, a larger value makes a more accurate model as well as
                                                                # larger inference latency and more memory

frontend_settings.filterbank_upper_band_limit = 7500.0
frontend_settings.filterbank_lower_band_limit = 125.0           # The dev board mic seems to have a lot of noise at lower frequencies

frontend_settings.noise_reduction_enable = True                 # Enable the noise reduction block to help ignore background noise in the field
frontend_settings.noise_reduction_smoothing_bits = 10
frontend_settings.noise_reduction_even_smoothing =  0.025
frontend_settings.noise_reduction_odd_smoothing = 0.06
frontend_settings.noise_reduction_min_signal_remaining = 0.40   # This value is fairly large (which makes the background noise reduction small)
                                                                # But it has been found to still give good results
                                                                # i.e. There is still some background noise reduction,
                                                                # but the actual signal is still (mostly) untouched

frontend_settings.dc_notch_filter_enable = True                 # Enable the DC notch filter, to help remove the DC signal from the dev board's mic
frontend_settings.dc_notch_filter_coefficient = 0.95

frontend_settings.quantize_dynamic_scale_enable = False         # We do NOT use dynamic quantization as we use sample-wise normalization instead


# Add the Audio Feature generator settings to the model parameters
# This way, they are included in the generated .tflite model file
# See https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html
my_model.model_parameters.update(frontend_settings)


# Set the sample-wise normalization setting.
# This tells the embedded audio frontend to do:
# spectrogram = (spectrogram - mean(spectrogram)) / std(spectrogram)
my_model.model_parameters['samplewise_norm.mean_and_std'] = True


##########################################################################################
# Specify the other dataset settings
#

my_model.input_shape = frontend_settings.spectrogram_shape + (1,)

# Add the keywords plus a _unknown_ meta class
my_model.classes = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    '_unknown_'
]
unknown_class_id = my_model.classes.index('_unknown_')

# Ensure the class weights are balanced during training
# https://towardsdatascience.com/why-weight-the-importance-of-training-on-balanced-datasets-f1e54688e7df
my_model.class_weights = 'balanced'



validation_split = 0.15

# Uncomment this to dump the augmented audio samples to the log directory
# DO NOT forget to disable this before training the model as it will generate A LOT of data
#data_dump_dir = my_model.create_log_dir('dataset_dump')

# This is the directory where the dataset will be extracted
dataset_dir = create_user_dir('datasets/keyword_spotting_numbers')


##########################################################################################
# Create the audio augmentation pipeline
#

# Install the other 3rd party packages required from preprocessing
install_pip_package('audiomentations')

import librosa
import audiomentations


def audio_pipeline_with_augmentations(
    path_batch:np.ndarray,
    label_batch:np.ndarray,
    seed:np.ndarray
) -> np.ndarray:
    """Augment a batch of audio clips and generate spectrograms

    This does the following, for each audio file path in the input batch:
    1. Read audio file
    2. Convolve the sample with a random impulse response
    3. For "unknown" samples, randomly replace with a cropped "known" sample
    4. Adjust its length to fit within the specified length
    5. Apply random augmentations to the audio sample using audiomentations
    6. Convert to the specified sample rate (if necessary)
    7. Generate a spectrogram from the augmented audio sample
    8. Dump the augmented audio and spectrogram (if necessary)

    NOTE: This will be execute in parallel across *separate* subprocesses.

    Arguments:
        path_batch: Batch of audio file paths
        label_batch: Batch of corresponding labels
        seed: Batch of seeds to use for random number generation,
            This ensures that the "random" augmentations are reproducible

    Return:
        Generated batch of spectrograms from augmented audio samples
    """
    batch_length = path_batch.shape[0]
    height, width = frontend_settings.spectrogram_shape
    x_shape = (batch_length, height, 1, width)
    x_batch = np.empty(x_shape, dtype=np.float32)

    # This is the amount of padding we add to the beginning of the sample
    # This allows for "warming up" the noise reduction block
    padding_length_ms = 1000
    padded_frontend_settings = frontend_settings.copy()
    padded_frontend_settings.sample_length_ms += padding_length_ms


    # Load the Impulse Response dataset into RAM once
    # and store in a global variable
    ir_dataset = globals().get('ir_dataset', None)
    if not ir_dataset:
        ir_dataset = audio_datasets.mit_ir_survey.load_dataset(f'{dataset_dir}/_ir_responses_')
        globals()['ir_dataset'] = ir_dataset


    # For each audio sample path in the current batch
    for i, (audio_path, labels) in enumerate(zip(path_batch, label_batch)):
        class_id = np.argmax(labels)
        rstate = np.random.RandomState(seed[i])
        rn = rstate.random()

        # For "unknown" samples,
        # Randomly convert a small percentage of them to "silence" or cropped "known" samples
        current_sample_is_in_unknown_class = class_id == unknown_class_id
        using_silence_as_unknown = current_sample_is_in_unknown_class and rn < 0.03
        use_cropped_sample_as_unknown = current_sample_is_in_unknown_class and not using_silence_as_unknown and rn < 0.15


        # If we should convert this "unknown" sample to silence
        if using_silence_as_unknown:
            original_sample_rate = frontend_settings.sample_rate_hz
            sample = np.zeros((original_sample_rate,), dtype=np.float32)
            audio_path = f'silence-{i}.wav'.encode('utf-8')

        # If we should convert his "unknown" sample to a cropped "known" sample
        elif use_cropped_sample_as_unknown:
            # Find a "known" sample in the current batch
            # Later, we'll crop this sample and use it as an "unknown" sample
            use_cropped_sample_as_unknown = False
            choices = list(range(batch_length))
            rstate.shuffle(choices)
            for choice_index in choices:
                if np.argmax(label_batch[choice_index]) == unknown_class_id:
                    continue

                audio_path = path_batch[choice_index]
                use_cropped_sample_as_unknown = True
                break

        # Read the audio file if it is not "silence"
        if not using_silence_as_unknown:
            try:
                sample, original_sample_rate = audio_utils.read_audio_file(audio_path, return_numpy=True, return_sample_rate=True)
            except Exception as e:
                raise RuntimeError(f'Failed to read: {audio_path}, err: {e}')

            # Applying an Impulse Response (IR)
            # This makes the sample sound like it was capture in a different environment
            # See https://siliconlabs.github.io/mltk/docs/python_api/datasets/audio/mit_ir_survey.html
            if len(sample) < original_sample_rate * 3.0 and rstate.random() < 0.80:
                sample = audio_datasets.mit_ir_survey.apply_random_ir(sample, ir_dataset, seed=seed[i])


        # Create a buffer to hold the padded sample
        padding_length = int((original_sample_rate * padding_length_ms) / 1000)
        padded_sample_length = int((original_sample_rate * padded_frontend_settings.sample_length_ms) / 1000)
        padded_sample = np.zeros((padded_sample_length,), dtype=np.float32)

        # If we want to crop a "known" sample and use it as an unknown sample
        if use_cropped_sample_as_unknown:
            audio_path = f'cropped-{i}.wav'.encode('utf-8')

            # Trim any silence from the "known" sample
            trimmed_sample, _ = librosa.effects.trim(sample, top_db=15)
            # Randomly insert a small percentage of the trimmed sample into padded sample buffer.
            # Note that the entire trimmed sample is actually added to the padded sample buffer
            # However, only the part of the sample that is after padding_length_ms will actually be used.
            # Everything before will eventually be dropped
            trimmed_sample_length = len(trimmed_sample)

            # Ensure the trimmed sample is no longer than 700ms
            if trimmed_sample_length < .7 * original_sample_rate:
                cropped_sample_percent = np.random.uniform(.20, .50)
                cropped_sample_length = int(trimmed_sample_length * cropped_sample_percent)
                # Add the beginning of the sample to the end of the padded sample buffer.
                # This simulates the sample streaming into the audio buffer,
                # but not being fully streamed in when an inference is invoked on the device.
                # In this case, we want the partial sample to be considered "unknown".
                padded_sample[-cropped_sample_length:] += trimmed_sample[:cropped_sample_length]


        else:
            # Adjust the audio clip to the length defined in the frontend_settings
            out_length = int((original_sample_rate * frontend_settings.sample_length_ms) / 1000)
            sample = audio_utils.adjust_length(
                sample,
                out_length=out_length,
                trim_threshold_db=30,
                offset=np.random.uniform(0, 1)
            )
            padded_sample[padding_length:padding_length+len(sample)] += sample



        # Initialize the global audio augmentations instance
        # NOTE: We want this to be global so that we only initialize it once per subprocess
        audio_augmentations = globals().get('audio_augmentations', None)
        if audio_augmentations is None:
            audio_augmentations = audiomentations.Compose(
                p=1.0,
                transforms=[
                audiomentations.Gain(min_gain_in_db=0.95, max_gain_in_db=1.2, p=1.0),
                audiomentations.AddBackgroundNoise(
                    f'{dataset_dir}/_background_noise_/ambient',
                    min_snr_in_db=-1, # The lower the SNR, the louder the background noise
                    max_snr_in_db=35,
                    noise_rms="relative",
                    lru_cache_size=50,
                    p=0.80
                ),
                audiomentations.AddBackgroundNoise(
                    f'{dataset_dir}/_background_noise_/brd2601',
                    min_absolute_rms_in_db=-75.0,
                    max_absolute_rms_in_db=-60.0,
                    noise_rms="absolute",
                    lru_cache_size=50,
                    p=1.0
                ),
                #audiomentations.AddGaussianSNR(min_snr_in_db=25, max_snr_in_db=40, p=0.25),
            ])
            globals()['audio_augmentations'] = audio_augmentations

        # Apply random augmentations to the audio sample
        augmented_sample = audio_augmentations(padded_sample, original_sample_rate)

        # Convert the sample rate (if necessary)
        if original_sample_rate != frontend_settings.sample_rate_hz:
            augmented_sample = audio_utils.resample(
                augmented_sample,
                orig_sr=original_sample_rate,
                target_sr=frontend_settings.sample_rate_hz
            )

        # Ensure the sample values are within (-1,1)
        augmented_sample = np.clip(augmented_sample, -1.0, 1.0)

        # Generate a spectrogram from the augmented audio sample
        spectrogram = audio_utils.apply_frontend(
            sample=augmented_sample,
            settings=padded_frontend_settings,
            dtype=np.uint16 # We just want the raw, uint16 output of the generated spectrogram
        )

        # The input audio sample was padded with padding_length_ms of background noise
        # Drop the padded background noise from the final spectrogram used for training
        spectrogram = spectrogram[-height:, :]

        # Normalize the spectrogram input about 0
        # spectrogram = (spectrogram - mean(spectrogram)) / std(spectrogram)
        # This is necessary to ensure the model is properly quantized
        # NOTE: The quantized .tflite will internally converted the float32 input to int8
        spectrogram = spectrogram.astype(np.float32)
        spectrogram -= np.mean(spectrogram, dtype=np.float32, keepdims=True)
        spectrogram /= (np.std(spectrogram, dtype=np.float32, keepdims=True) + 1e-6)

        # Convert the spectrogram dimension from:
        # <time, features> to <time, 1, features>
        # as this is the input shape the TENet model architecture expects
        spectrogram = np.expand_dims(spectrogram, axis=-2)

        x_batch[i] = spectrogram

        # Dump the augmented audio sample AND corresponding spectrogram (if necessary)
        data_dump_dir = globals().get('data_dump_dir', None)
        if data_dump_dir:
            try:
                from cv2 import cv2
            except:
                import cv2

            fn = os.path.basename(audio_path.decode('utf-8'))

            audio_dump_path = f'{data_dump_dir}/{class_id}-{fn[:-4]}-{seed[0]}.wav'
            spectrogram_dumped = np.squeeze(spectrogram, axis=-2) # Convert to <time, features>
            # Transpose to put the time on the x-axis
            spectrogram_dumped = np.transpose(spectrogram_dumped)

            # Convert from float32 to uint8
            # NOTE: We hardcode the min/max values so that the spectrogram images are consistent..
            max_val = 2 # np.max(spectrogram_dumped)
            min_val = -2 # np.min(spectrogram_dumped)
            val_range = max_val - min_val
            spectrogram_dumped = (spectrogram_dumped - min_val) * 255 / val_range
            spectrogram_dumped = np.clip(spectrogram_dumped, 0, 255)
            spectrogram_dumped = spectrogram_dumped.astype(np.uint8)
            # Increase the size of the spectrogram to make it easier to see as a jpeg
            spectrogram_dumped = cv2.resize(spectrogram_dumped, (height*3,width*3))

            valid_sample_length = int((frontend_settings.sample_length_ms * frontend_settings.sample_rate_hz) / 1000)
            valid_augmented_sample = augmented_sample[-valid_sample_length:]
            audio_dump_path = audio_utils.write_audio_file(
                audio_dump_path,
                valid_augmented_sample,
                sample_rate=frontend_settings.sample_rate_hz
            )
            image_dump_path = audio_dump_path.replace('.wav', '.jpg')
            jpg_data = cv2.applyColorMap(spectrogram_dumped, cv2.COLORMAP_HOT)
            cv2.imwrite(image_dump_path, jpg_data)


    return x_batch


##########################################################################################
# Define the MltkDataset object
# NOTE: This class is optional but is useful for organizing the code
#
class MyDataset(mltk_core.MltkDataset):

    def __init__(self):
        super().__init__()
        self.pools = []
        self.summary = ''

    def summarize_dataset(self) -> str:
        """Return a string summary of the dataset"""
        s = self.summary
        s += mltk_core.MltkDataset.summarize_class_counts(my_model.class_counts)
        return s


    def load_dataset(
        self,
        subset: str,
        test:bool = False,
        **kwargs
    ) -> Tuple[tf.data.Dataset, None, tf.data.Dataset]:
        """Load the dataset subset

        This is called automatically by the MLTK before training
        or evaluation.

        Args:
            subset: The dataset subset to return: 'training' or 'evaluation'
            test: This is optional, it is used when invoking a training "dryrun", e.g.: mltk train audio_tf_dataset-test
                If this is true, then only return a small portion of the dataset for testing purposes

        Return:
            if subset == training:
                A tuple, (train_dataset, None, validation_dataset)
            else:
                validation_dataset
        """

        if subset == 'training':
            x = self.load_subset('training', test=test)
            validation_data = self.load_subset('validation', test=test)

            return x, None, validation_data

        else:
            x = self.load_subset('validation', test=test)
            return x

    def unload_dataset(self):
        """Unload the dataset by shutting down the processing pools"""
        for pool in self.pools:
            pool.shutdown()
        self.pools.clear()


    def load_subset(self, subset:str, test:bool) -> tf.data.Dataset:
        """Load the subset"""
        if subset in ('validation', 'evaluation'):
            split = (0, validation_split)
        elif subset == 'training':
            split = (validation_split, 1)
            data_dump_dir = globals().get('data_dump_dir', None)
            if data_dump_dir:
                print(f'\n\n*** Dumping augmented samples to: {data_dump_dir}\n\n')
        else:
            split = None
            my_model.class_counts = {}


        # Download the synthetic "ten_digits" dataset and extract into the dataset directory
        audio_datasets.ten_digits.download(dataset_dir, clean_dest_dir=True)
        # Download the Google speech commands dataset into the dataset directory
        # This effectively combines the two datasets
        audio_datasets.speech_commands_v2.load_clean_data(dataset_dir, clean_dest_dir=False)

        # Download the mlcommons subset and extract into the dataset sub-directory: '_unknown/mlcommons_keywords'
        audio_datasets.mlcommons.ml_commons_keywords.download(f'{dataset_dir}/_unknown/mlcommons_keywords')

        # Download the mlcommons ESC-50 dataset and extract into the dataset sub-directory: '_unknown/esc-50'
        audio_datasets.background_noise.esc50.download(f'{dataset_dir}/_unknown/esc-50', sample_rate_hertz=frontend_settings.sample_rate_hz)

        # Download the MIT Impulse Response dataset into into the dataset sub-directory: '_ir_responses_'
        audio_datasets.mit_ir_survey.download(f'{dataset_dir}/_ir_responses_', sample_rate_hz=frontend_settings.sample_rate_hz)

        # Download the BRD2601 background microphone audio and add it to the _background_noise_/brd2601 of the dataset
        audio_datasets.background_noise.brd2601.download(f'{dataset_dir}/_background_noise_/brd2601', sample_rate_hertz=frontend_settings.sample_rate_hz)

        # Download other ambient background audio and add it to the _background_noise_/ambient of the dataset
        audio_datasets.background_noise.ambient.download(
            f'{dataset_dir}/_background_noise_/ambient',
            sample_rate_hertz=frontend_settings.sample_rate_hz
        )


        # Create a tf.data.Dataset from the extracted dataset directory
        max_samples_per_class = my_model.batch_size if test else -1
        class_counts = my_model.class_counts[subset] if subset else my_model.class_counts
        features_ds, labels_ds = tf_dataset_utils.load_audio_directory(
            directory=dataset_dir,
            classes=my_model.classes,
            onehot_encode=True, # We're using categorical cross-entropy so one-hot encode the labels
            shuffle=True,
            seed=42,
            max_samples_per_class=max_samples_per_class,
            unknown_class_percentage=0, # We manually populate the "known" class in the add_unknown_samples() callback
            split=split,
            return_audio_data=False, # We only want to return the file paths
            class_counts=class_counts,
            list_valid_filenames_in_directory_function=self.list_valid_filenames_in_directory,
            process_samples_function=self.add_unknown_samples
        )

        if subset:
            # The number of batches to process in each subprocess
            per_job_batch_multiplier = 1000
            per_job_batch_size = my_model.batch_size * per_job_batch_multiplier

            # We use an incrementing counter as the seed for the random augmentations
            # This helps to keep the training reproducible
            try:
                seed_counter = tf.data.Dataset.counter()
            except:
                seed_counter = tf.data.experimental.Counter()
            features_ds = features_ds.zip((features_ds, labels_ds, seed_counter))

            # Usage of tf_dataset_utils.parallel_process()
            # is optional, but can speed-up training as the data augmentations
            # are spread across the available CPU cores.
            # Each CPU core gets its own subprocess,
            # and and subprocess executes audio_augmentation_pipeline() on batches of the dataset.

            features_ds = features_ds.batch(per_job_batch_size // per_job_batch_multiplier, drop_remainder=True)
            labels_ds = labels_ds.batch(per_job_batch_size // per_job_batch_multiplier, drop_remainder=True)
            features_ds, pool = tf_dataset_utils.parallel_process(
                features_ds,
                audio_pipeline_with_augmentations,
                dtype=np.float32,
                #n_jobs=84 if subset == 'training' else 32, # These are the settings for a 256 CPU core cloud machine
                #72 if subset == 'training' else 32, # These are the settings for a 128 CPU core cloud machine
                #n_jobs=44 if subset == 'training' else 16, # These are the settings for a 96 CPU core cloud machine
                #n_jobs=50 if subset == 'training' else 25, # These are the settings for a 84 CPU core cloud machine
                #n_jobs=36 if subset == 'training' else 12, # These are the settings for a 64 CPU core cloud machine
                #n_jobs=28 if subset == 'training' else 16, # These are the settings for a 48 CPU core cloud machine
                #n_jobs=.65 if subset == 'training' else .35,
                n_jobs=8,
                name=subset,
            )
            self.pools.append(pool)
            features_ds = features_ds.unbatch()
            labels_ds = labels_ds.unbatch()

            # Pre-fetching batches can help with throughput
            features_ds = features_ds.prefetch(per_job_batch_size)

        # Combine the augmented audio samples with their corresponding labels
        ds = tf.data.Dataset.zip((features_ds, labels_ds))

        # Shuffle the data for each sample
        # A perfect shuffle would use n_samples but this can slow down training,
        # so we just shuffle batches of the data
        #ds = ds.shuffle(n_samples, reshuffle_each_iteration=True)
        if not test:
            ds = ds.shuffle(per_job_batch_size, reshuffle_each_iteration=True)

        # At this point we have a flat dataset of x,y tuples
        # Batch the data as necessary for training
        ds = ds.batch(my_model.batch_size)

        # Pre-fetch a couple training batches to aid throughput
        ds = ds.prefetch(2)

        return ds

    def list_valid_filenames_in_directory(
        self,
        base_directory:str,
        search_class:str,
        white_list_formats:List[str],
        split:float,
        follow_links:bool,
        shuffle_index_directory:str
    ) -> Tuple[str, List[str]]:
        """Return a list of valid file names for the given class

        This is called by the tf_dataset_utils.load_audio_directory() API.

        # This uses shuffle_file_list_by_group() helper function so that the same "voices"
        # are only present in a particular subset.
        """
        assert shuffle_index_directory is None, 'Shuffling the index is not supported by this dataset'

        file_list = []
        index_path = f'{base_directory}/.index/{search_class}.txt'

        # If the index file exists, then read it
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                for line in f:
                    file_list.append(line.strip())

        else:
            # Else find all files for the given class in the search directory
            class_base_dir = f'{base_directory}/{search_class}/'
            for root, _, files in os.walk(base_directory, followlinks=follow_links):
                root = root.replace('\\', '/') + '/'
                if not root.startswith(class_base_dir):
                    continue

                for fname in files:
                    if not fname.lower().endswith(white_list_formats):
                        continue
                    abs_path = os.path.join(root, fname)
                    if os.path.getsize(abs_path) == 0:
                        continue
                    rel_path = os.path.relpath(abs_path, base_directory)
                    file_list.append(rel_path.replace('\\', '/'))


                # Shuffle the voice groups
                # then flatten into list
                # This way, when the list is split into training and validation sets
                # the same voice only appears in one subset
                file_list = shuffle_file_list_by_group(file_list, get_sample_group_id_from_path)

                # Write the file list file
                mltk_core.get_mltk_logger().info(f'Generating index for "{search_class}" ({len(file_list)} samples): {index_path}')
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                with open(index_path, 'w') as f:
                    for p in file_list:
                        f.write(p + '\n')

        if len(file_list) == 0:
            raise RuntimeError(f'No samples found for class: {search_class}')


        n_files = len(file_list)
        if split[0] == 0:
            start = 0
            stop = math.ceil(split[1] * n_files)

            # We want to ensure the same person isn't in both subsets
            # So, ensure that the split point does NOT
            # split with file names with the same hash
            # recall: same hash = same person saying word

            # Get the hash of the other subset
            other_subset_hash = get_sample_group_id_from_path(file_list[stop])
            # Keep moving the 'stop' index back while
            # it's index matches the otherside
            while stop > 0 and get_sample_group_id_from_path(file_list[stop-1]) == other_subset_hash:
                stop -= 1

        else:
            start = math.ceil(split[0] * n_files)
            # Get the hash of the this subset
            this_subset_hash = get_sample_group_id_from_path(file_list[start])
            # Keep moving the 'start' index back while
            # it's index matches this side's
            while start > 0 and get_sample_group_id_from_path(file_list[start-1]) == this_subset_hash:
                start -= 1

            stop = n_files

        filenames = file_list[start:stop]

        return search_class, filenames

    def add_unknown_samples(
        self,
        directory:str,
        sample_paths:Dict[str,str], # A dictionary: <class name>, [<sample paths relative to directory>],
        split:Tuple[float,float],
        follow_links:bool,
        white_list_formats:List[str],
        shuffle:bool,
        seed:int,
        **kwargs
    ):
        """Generate a list of all possible "unknown" samples for this given subset.

        Then populate the "_unknown_" class with a random subset of the "unknown" samples.
        The subset should be the approximate size of the "known" samples

        """
        mlcommons_keywords_dir = f'{dataset_dir}/_unknown/mlcommons_keywords'
        esc50_dir = f'{dataset_dir}/_unknown/esc-50/audio'

        # Create a list of all possible "unknown" samples
        file_list = []

        # All all the mlcommons_keywords "unknown" samples that are not the "known" sample
        all_keywords = []
        for kw in os.listdir(mlcommons_keywords_dir):
            if kw in my_model.classes:
                continue
            d = f'{mlcommons_keywords_dir}/{kw}'
            if not os.path.isdir(d):
                continue

            for fn in os.listdir(d):
                if fn.endswith('.wav'):
                    all_keywords.append(f'_unknown/mlcommons_keywords/{kw}/{fn}')

        # Get a random subset of the "unknown" samples
        # We only select 11k so balance with the "known" classes
        rng = np.random.RandomState(seed)
        all_keywords = sorted(all_keywords)
        rng.shuffle(all_keywords)
        file_list.extend(all_keywords[:11000])

        # Add all the samples from the ESC-50 dataset which is 2k samples
        # This way, we have random keywords and random noises in the "unknown" class's sample list
        for fn in os.listdir(esc50_dir):
            if not fn.endswith('.wav'):
                continue
            file_list.append(f'_unknown/esc-50/audio/{fn}')

        # Sort the unknown samples by "voice"
        # This helps to ensure voices are only present in a given subset
        file_list = sorted(file_list)
        file_list = shuffle_file_list_by_group(file_list, get_sample_group_id_from_path)

        # Split the file list for the current subset
        sample_paths['_unknown_'] = split_file_list(file_list, split)



def get_sample_group_id_from_path(p:str) -> str:
    """Extract the "voice hash" from the sample path.

    This is used by shuffle_file_list_by_group() so that when we split
    the dataset for training and validation, the same "voice" only appears
    in one of the subsets.
    """
    fn = os.path.basename(p)
    fn = fn.replace('.wav', '').replace('.mp3', '')

    # If this sample is from the Google speech commands dataset
    #  c53b335a_nohash_1.wav -> c53b335a
    if '_nohash_' in fn:
        toks = fn.split('_')
        return toks[0]

    # If this sample is from an mlcommons dataset
    #  common_voice_en_20127845.wav -> 20127845
    if fn.startswith('common_voice_'):
        toks = fn.split('_')
        return toks[-1]

    # If this sample is from a silabs synthetic dataset
    # azure_af-ZA+AdriNeural+None+aww+medium+low+588b6ace.wav -> 588b6ace
    if fn.startswith(('gcp_', 'azure_', 'aws_')):
        toks = fn.split('+')
        return toks[-1]

    if '/esc-50/' in p:
        toks = fn.split('-')
        return toks[1]

    raise RuntimeError(f'Failed to get voice hash from {p}')


my_model.dataset = MyDataset()


my_model.model_parameters['average_window_duration_ms'] = 450
# Define a specific detection threshold for each class
my_model.model_parameters['detection_threshold'] = int(0.95*255)
#my_model.model_parameters['detection_threshold_list'] = list(map(lambda x: int(x*255), [.95, .95, .95, .95,  1.0]))
# Amount of milliseconds to wait after a keyword is detected before detecting the SAME keyword again
# A different keyword may be detected immediately after
my_model.model_parameters['suppression_ms'] = 700
# The minimum number of inference results to average when calculating the detection value
my_model.model_parameters['minimum_count'] = 2
# Set the volume gain scaler (i.e. amplitude) to apply to the microphone data. If 0 or omitted, no scaler is applied
my_model.model_parameters['volume_gain'] = 0.0
# This the amount of time in milliseconds between audio processing loops
# Since we're using the audio detection block, we want this to be as short as possible
my_model.model_parameters['latency_ms'] = 10
# Enable verbose inference results
my_model.model_parameters['verbose_model_output_logs'] = False


##########################################################################################
# The following allows for running this model training script directly, e.g.:
# python keyword_spotting_numbers.py
#
# Note that this has the same functionality as:
# mltk train keyword_spotting_numbers
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=True)


    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command:  mltk train keyword_spotting_numbers-test --clean)
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate keyword_spotting_numbers-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile keyword_spotting_numbers-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)
