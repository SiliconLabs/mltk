"""keyword_spotting_with_transfer_learning
********************************************

- Source code: `keyword_spotting_with_transfer_learning.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_with_transfer_learning.py>`_
- Pre-trained model: `keyword_spotting_with_transfer_learning.mltk.zip <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_with_transfer_learning.mltk.zip>`_


This model specification script is designed to work with the
`Keyword Spotting with Transfer Learning <https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_with_transfer_learning.html>`_ tutorial.

This model uses the weights from the previously trained model `keyword_spotting_mobilenetv2.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_mobilenetv2.py>`_,
as a starting point for training this model to detect the keywords:

- one
- two
- three
- four
- five
- six
- seven
- eight
- nine


This process is known as `Transfer Learning <https://en.wikipedia.org/wiki/Transfer_learning>`_ and can greatly improve training times as
the new model can leverage the knowledge learnt from another model.

See the `Keras Documentation <https://keras.io/guides/transfer_learning>`_ for more details about how to use
the Keras API to enable transfer learning.

Dataset
---------
This uses the :py:class:`mltk.datasets.audio.speech_commands.speech_commands_v2` dataset provided by Google.

Preprocessing
--------------
This uses the :py:class:`mltk.core.preprocess.audio.parallel_generator.ParallelAudioDataGenerator` with the
:py:class:`mltk.core.preprocess.audio.audio_feature_generator.AudioFeatureGenerator` settings:

- sample_rate: 16kHz
- sample_length: 1.2s
- window size: 30ms
- window step: 20ms
- n_channels: 49


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train keyword_spotting_with_transfer_learning-test

   # Train the model
   mltk train keyword_spotting_with_transfer_learning

   # Evaluate the trained model .tflite model
   mltk evaluate keyword_spotting_with_transfer_learning --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile keyword_spotting_with_transfer_learning --accelerator MVP

   # Profile the model on a physical development board
   mltk profile keyword_spotting_with_transfer_learning --accelerator MVP --device

   # Run the model in the audio classifier on the local PC
   mltk classify_audio keyword_spotting_with_transfer_learning --verbose

   # Run the model in the audio classifier on the physical device
   mltk classify_audio keyword_spotting_with_transfer_learning --device --verbose


Model Summary
--------------

.. code-block:: shell

    mltk summarize keyword_spotting_with_transfer_learning --tflite

    +-------+-------------------+-----------------+-----------------+-------------------------------------------------------+
    | Index | OpCode            | Input(s)        | Output(s)       | Config                                                |
    +-------+-------------------+-----------------+-----------------+-------------------------------------------------------+
    | 0     | conv_2d           | 59x49x1 (int8)  | 30x25x8 (int8)  | Padding:same stride:2x2 activation:relu6              |
    |       |                   | 3x3x1 (int8)    |                 |                                                       |
    |       |                   | 8 (int32)       |                 |                                                       |
    | 1     | depthwise_conv_2d | 30x25x8 (int8)  | 30x25x8 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x8 (int8)    |                 |                                                       |
    |       |                   | 8 (int32)       |                 |                                                       |
    | 2     | conv_2d           | 30x25x8 (int8)  | 30x25x8 (int8)  | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x8 (int8)    |                 |                                                       |
    |       |                   | 8 (int32)       |                 |                                                       |
    | 3     | add               | 30x25x8 (int8)  | 30x25x8 (int8)  | Activation:none                                       |
    |       |                   | 30x25x8 (int8)  |                 |                                                       |
    | 4     | conv_2d           | 30x25x8 (int8)  | 30x25x48 (int8) | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x8 (int8)    |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 5     | pad               | 30x25x48 (int8) | 31x27x48 (int8) | BuiltinOptionsType=22                                 |
    |       |                   | 2 (int32)       |                 |                                                       |
    | 6     | depthwise_conv_2d | 31x27x48 (int8) | 15x13x48 (int8) | Multipler:1 padding:valid stride:2x2 activation:relu6 |
    |       |                   | 3x3x48 (int8)   |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 7     | conv_2d           | 15x13x48 (int8) | 15x13x8 (int8)  | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x48 (int8)   |                 |                                                       |
    |       |                   | 8 (int32)       |                 |                                                       |
    | 8     | conv_2d           | 15x13x8 (int8)  | 15x13x48 (int8) | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x8 (int8)    |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 9     | depthwise_conv_2d | 15x13x48 (int8) | 15x13x48 (int8) | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x48 (int8)   |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 10    | conv_2d           | 15x13x48 (int8) | 15x13x8 (int8)  | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x48 (int8)   |                 |                                                       |
    |       |                   | 8 (int32)       |                 |                                                       |
    | 11    | add               | 15x13x8 (int8)  | 15x13x8 (int8)  | Activation:none                                       |
    |       |                   | 15x13x8 (int8)  |                 |                                                       |
    | 12    | conv_2d           | 15x13x8 (int8)  | 15x13x48 (int8) | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x8 (int8)    |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 13    | pad               | 15x13x48 (int8) | 17x15x48 (int8) | BuiltinOptionsType=22                                 |
    |       |                   | 2 (int32)       |                 |                                                       |
    | 14    | depthwise_conv_2d | 17x15x48 (int8) | 8x7x48 (int8)   | Multipler:1 padding:valid stride:2x2 activation:relu6 |
    |       |                   | 3x3x48 (int8)   |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 15    | conv_2d           | 8x7x48 (int8)   | 8x7x8 (int8)    | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x48 (int8)   |                 |                                                       |
    |       |                   | 8 (int32)       |                 |                                                       |
    | 16    | conv_2d           | 8x7x8 (int8)    | 8x7x48 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x8 (int8)    |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 17    | depthwise_conv_2d | 8x7x48 (int8)   | 8x7x48 (int8)   | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x48 (int8)   |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 18    | conv_2d           | 8x7x48 (int8)   | 8x7x8 (int8)    | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x48 (int8)   |                 |                                                       |
    |       |                   | 8 (int32)       |                 |                                                       |
    | 19    | add               | 8x7x8 (int8)    | 8x7x8 (int8)    | Activation:none                                       |
    |       |                   | 8x7x8 (int8)    |                 |                                                       |
    | 20    | conv_2d           | 8x7x8 (int8)    | 8x7x48 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x8 (int8)    |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 21    | depthwise_conv_2d | 8x7x48 (int8)   | 8x7x48 (int8)   | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x48 (int8)   |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 22    | conv_2d           | 8x7x48 (int8)   | 8x7x8 (int8)    | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x48 (int8)   |                 |                                                       |
    |       |                   | 8 (int32)       |                 |                                                       |
    | 23    | add               | 8x7x8 (int8)    | 8x7x8 (int8)    | Activation:none                                       |
    |       |                   | 8x7x8 (int8)    |                 |                                                       |
    | 24    | conv_2d           | 8x7x8 (int8)    | 8x7x48 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x8 (int8)    |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 25    | pad               | 8x7x48 (int8)   | 9x9x48 (int8)   | BuiltinOptionsType=22                                 |
    |       |                   | 2 (int32)       |                 |                                                       |
    | 26    | depthwise_conv_2d | 9x9x48 (int8)   | 4x4x48 (int8)   | Multipler:1 padding:valid stride:2x2 activation:relu6 |
    |       |                   | 3x3x48 (int8)   |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 27    | conv_2d           | 4x4x48 (int8)   | 4x4x16 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x48 (int8)   |                 |                                                       |
    |       |                   | 16 (int32)      |                 |                                                       |
    | 28    | conv_2d           | 4x4x16 (int8)   | 4x4x96 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x16 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 29    | depthwise_conv_2d | 4x4x96 (int8)   | 4x4x96 (int8)   | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x96 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 30    | conv_2d           | 4x4x96 (int8)   | 4x4x16 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x96 (int8)   |                 |                                                       |
    |       |                   | 16 (int32)      |                 |                                                       |
    | 31    | add               | 4x4x16 (int8)   | 4x4x16 (int8)   | Activation:none                                       |
    |       |                   | 4x4x16 (int8)   |                 |                                                       |
    | 32    | conv_2d           | 4x4x16 (int8)   | 4x4x96 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x16 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 33    | depthwise_conv_2d | 4x4x96 (int8)   | 4x4x96 (int8)   | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x96 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 34    | conv_2d           | 4x4x96 (int8)   | 4x4x16 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x96 (int8)   |                 |                                                       |
    |       |                   | 16 (int32)      |                 |                                                       |
    | 35    | add               | 4x4x16 (int8)   | 4x4x16 (int8)   | Activation:none                                       |
    |       |                   | 4x4x16 (int8)   |                 |                                                       |
    | 36    | conv_2d           | 4x4x16 (int8)   | 4x4x96 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x16 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 37    | depthwise_conv_2d | 4x4x96 (int8)   | 4x4x96 (int8)   | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x96 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 38    | conv_2d           | 4x4x96 (int8)   | 4x4x16 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x96 (int8)   |                 |                                                       |
    |       |                   | 16 (int32)      |                 |                                                       |
    | 39    | add               | 4x4x16 (int8)   | 4x4x16 (int8)   | Activation:none                                       |
    |       |                   | 4x4x16 (int8)   |                 |                                                       |
    | 40    | conv_2d           | 4x4x16 (int8)   | 4x4x96 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x16 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 41    | depthwise_conv_2d | 4x4x96 (int8)   | 4x4x96 (int8)   | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x96 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 42    | conv_2d           | 4x4x96 (int8)   | 4x4x16 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x96 (int8)   |                 |                                                       |
    |       |                   | 16 (int32)      |                 |                                                       |
    | 43    | add               | 4x4x16 (int8)   | 4x4x16 (int8)   | Activation:none                                       |
    |       |                   | 4x4x16 (int8)   |                 |                                                       |
    | 44    | conv_2d           | 4x4x16 (int8)   | 4x4x96 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x16 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 45    | depthwise_conv_2d | 4x4x96 (int8)   | 4x4x96 (int8)   | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x96 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 46    | conv_2d           | 4x4x96 (int8)   | 4x4x16 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x96 (int8)   |                 |                                                       |
    |       |                   | 16 (int32)      |                 |                                                       |
    | 47    | add               | 4x4x16 (int8)   | 4x4x16 (int8)   | Activation:none                                       |
    |       |                   | 4x4x16 (int8)   |                 |                                                       |
    | 48    | conv_2d           | 4x4x16 (int8)   | 4x4x96 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x16 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 49    | depthwise_conv_2d | 4x4x96 (int8)   | 4x4x96 (int8)   | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x96 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 50    | conv_2d           | 4x4x96 (int8)   | 4x4x16 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x96 (int8)   |                 |                                                       |
    |       |                   | 16 (int32)      |                 |                                                       |
    | 51    | add               | 4x4x16 (int8)   | 4x4x16 (int8)   | Activation:none                                       |
    |       |                   | 4x4x16 (int8)   |                 |                                                       |
    | 52    | conv_2d           | 4x4x16 (int8)   | 4x4x96 (int8)   | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x16 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 53    | pad               | 4x4x96 (int8)   | 5x5x96 (int8)   | BuiltinOptionsType=22                                 |
    |       |                   | 2 (int32)       |                 |                                                       |
    | 54    | depthwise_conv_2d | 5x5x96 (int8)   | 2x2x96 (int8)   | Multipler:1 padding:valid stride:2x2 activation:relu6 |
    |       |                   | 3x3x96 (int8)   |                 |                                                       |
    |       |                   | 96 (int32)      |                 |                                                       |
    | 55    | conv_2d           | 2x2x96 (int8)   | 2x2x24 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x96 (int8)   |                 |                                                       |
    |       |                   | 24 (int32)      |                 |                                                       |
    | 56    | conv_2d           | 2x2x24 (int8)   | 2x2x144 (int8)  | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x24 (int8)   |                 |                                                       |
    |       |                   | 144 (int32)     |                 |                                                       |
    | 57    | depthwise_conv_2d | 2x2x144 (int8)  | 2x2x144 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x144 (int8)  |                 |                                                       |
    |       |                   | 144 (int32)     |                 |                                                       |
    | 58    | conv_2d           | 2x2x144 (int8)  | 2x2x24 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x144 (int8)  |                 |                                                       |
    |       |                   | 24 (int32)      |                 |                                                       |
    | 59    | add               | 2x2x24 (int8)   | 2x2x24 (int8)   | Activation:none                                       |
    |       |                   | 2x2x24 (int8)   |                 |                                                       |
    | 60    | conv_2d           | 2x2x24 (int8)   | 2x2x144 (int8)  | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x24 (int8)   |                 |                                                       |
    |       |                   | 144 (int32)     |                 |                                                       |
    | 61    | depthwise_conv_2d | 2x2x144 (int8)  | 2x2x144 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x144 (int8)  |                 |                                                       |
    |       |                   | 144 (int32)     |                 |                                                       |
    | 62    | conv_2d           | 2x2x144 (int8)  | 2x2x24 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x144 (int8)  |                 |                                                       |
    |       |                   | 24 (int32)      |                 |                                                       |
    | 63    | add               | 2x2x24 (int8)   | 2x2x24 (int8)   | Activation:none                                       |
    |       |                   | 2x2x24 (int8)   |                 |                                                       |
    | 64    | conv_2d           | 2x2x24 (int8)   | 2x2x144 (int8)  | Padding:same stride:1x1 activation:relu6              |
    |       |                   | 1x1x24 (int8)   |                 |                                                       |
    |       |                   | 144 (int32)     |                 |                                                       |
    | 65    | depthwise_conv_2d | 2x2x144 (int8)  | 2x2x144 (int8)  | Multipler:1 padding:same stride:1x1 activation:relu6  |
    |       |                   | 3x3x144 (int8)  |                 |                                                       |
    |       |                   | 144 (int32)     |                 |                                                       |
    | 66    | conv_2d           | 2x2x144 (int8)  | 2x2x48 (int8)   | Padding:same stride:1x1 activation:none               |
    |       |                   | 1x1x144 (int8)  |                 |                                                       |
    |       |                   | 48 (int32)      |                 |                                                       |
    | 67    | conv_2d           | 2x2x48 (int8)   | 2x2x384 (int8)  | Padding:valid stride:1x1 activation:relu6             |
    |       |                   | 1x1x48 (int8)   |                 |                                                       |
    |       |                   | 384 (int32)     |                 |                                                       |
    | 68    | mean              | 2x2x384 (int8)  | 384 (int8)      | BuiltinOptionsType=27                                 |
    |       |                   | 2 (int32)       |                 |                                                       |
    | 69    | fully_connected   | 384 (int8)      | 11 (int8)       | Activation:none                                       |
    |       |                   | 384 (int8)      |                 |                                                       |
    |       |                   | 11 (int32)      |                 |                                                       |
    | 70    | softmax           | 11 (int8)       | 11 (int8)       | BuiltinOptionsType=9                                  |
    +-------+-------------------+-----------------+-----------------+-------------------------------------------------------+
    Total MACs: 1.738 M
    Total OPs: 3.979 M
    Name: keyword_spotting_with_transfer_learning
    Version: 1
    Description: Keyword spotting classifier using transfer learning to detect: "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
    Classes: one, two, three, four, five, six, seven, eight, nine, _unknown_, _silence_
    hash: 536b7f0ff2de1d1db7c6b35d1e8cb721
    date: 2022-02-04T19:07:10.433Z
    runtime_memory_size: 105032
    average_window_duration_ms: 1000
    detection_threshold: 145
    suppression_ms: 1000
    minimum_count: 3
    volume_db: 5.0
    latency_ms: 0
    log_level: info
    samplewise_norm.rescale: 0.0
    samplewise_norm.mean_and_std: False
    fe.sample_rate_hz: 16000
    fe.sample_length_ms: 1200
    fe.window_size_ms: 30
    fe.window_step_ms: 20
    fe.filterbank_n_channels: 49
    fe.filterbank_upper_band_limit: 3999.0
    fe.filterbank_lower_band_limit: 125.0
    fe.noise_reduction_enable: True
    fe.noise_reduction_smoothing_bits: 10
    fe.noise_reduction_even_smoothing: 0.02500000037252903
    fe.noise_reduction_odd_smoothing: 0.05999999865889549
    fe.noise_reduction_min_signal_remaining: 0.029999999329447746
    fe.pcan_enable: False
    fe.pcan_strength: 0.949999988079071
    fe.pcan_offset: 80.0
    fe.pcan_gain_bits: 21
    fe.log_scale_enable: True
    fe.log_scale_shift: 6
    fe.fft_length: 512
    .tflite file size: 260.2kB



Model Profiling Report
-----------------------

.. code-block:: shell

   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile keyword_spotting_with_transfer_learning --device --accelerator MVP

    Profiling Summary
    Name: keyword_spotting_with_transfer_learning
    Accelerator: MVP
    Input Shape: 1x59x49x1
    Input Data Type: int8
    Output Shape: 1x11
    Output Data Type: int8
    Flash, Model File Size (bytes): 260.2k
    RAM, Runtime Memory Size (bytes): 104.9k
    Operation Count: 4.2M
    Multiply-Accumulate Count: 1.7M
    Layer Count: 71
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 2.2M
    CPU Cycle Count: 3.1M
    CPU Utilization (%): 67.9
    Clock Rate (hz): 78.0M
    Time (s): 58.4m
    Ops/s: 72.6M
    MACs/s: 29.8M
    Inference/s: 17.1

    Model Layers
    +-------+-------------------+--------+--------+------------+------------+----------+-------------------------+--------------+--------------------------------------------------------+
    | Index | OpCode            | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape             | Output Shape | Options                                                |
    +-------+-------------------+--------+--------+------------+------------+----------+-------------------------+--------------+--------------------------------------------------------+
    | 0     | conv_2d           | 126.0k | 54.0k  | 86.7k      | 27.9k      | 1.3m     | 1x59x49x1,8x3x3x1,8     | 1x30x25x8    | Padding:same stride:2x2 activation:relu6               |
    | 1     | depthwise_conv_2d | 126.0k | 54.0k  | 189.4k     | 214.1k     | 4.4m     | 1x30x25x8,1x3x3x8,8     | 1x30x25x8    | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 2     | conv_2d           | 102.0k | 48.0k  | 57.7k      | 5.9k       | 780.0u   | 1x30x25x8,8x1x1x8,8     | 1x30x25x8    | Padding:same stride:1x1 activation:none                |
    | 3     | add               | 6.0k   | 0      | 21.0k      | 4.6k       | 330.0u   | 1x30x25x8,1x30x25x8     | 1x30x25x8    | Activation:none                                        |
    | 4     | conv_2d           | 684.0k | 288.0k | 327.7k     | 5.3k       | 4.2m     | 1x30x25x8,48x1x1x8,48   | 1x30x25x48   | Padding:same stride:1x1 activation:relu6               |
    | 5     | pad               | 241.1k | 0      | 0          | 967.9k     | 12.2m    | 1x30x25x48,4x2          | 1x31x27x48   | Type=padoptions                                        |
    | 6     | depthwise_conv_2d | 196.6k | 84.2k  | 157.8k     | 282.4k     | 3.6m     | 1x31x27x48,1x3x3x48,48  | 1x15x13x48   | Multiplier:1 padding:valid stride:2x2 activation:relu6 |
    | 7     | conv_2d           | 151.3k | 74.9k  | 65.6k      | 5.5k       | 870.0u   | 1x15x13x48,8x1x1x48,8   | 1x15x13x8    | Padding:same stride:1x1 activation:none                |
    | 8     | conv_2d           | 177.8k | 74.9k  | 85.1k      | 5.3k       | 1.1m     | 1x15x13x8,48x1x1x8,48   | 1x15x13x48   | Padding:same stride:1x1 activation:relu6               |
    | 9     | depthwise_conv_2d | 196.6k | 84.2k  | 145.4k     | 280.3k     | 3.6m     | 1x15x13x48,1x3x3x48,48  | 1x15x13x48   | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 10    | conv_2d           | 151.3k | 74.9k  | 65.8k      | 5.4k       | 870.0u   | 1x15x13x48,8x1x1x48,8   | 1x15x13x8    | Padding:same stride:1x1 activation:none                |
    | 11    | add               | 1.6k   | 0      | 5.5k       | 2.7k       | 90.0u    | 1x15x13x8,1x15x13x8     | 1x15x13x8    | Activation:none                                        |
    | 12    | conv_2d           | 177.8k | 74.9k  | 85.5k      | 5.3k       | 1.1m     | 1x15x13x8,48x1x1x8,48   | 1x15x13x48   | Padding:same stride:1x1 activation:relu6               |
    | 13    | pad               | 73.4k  | 0      | 0          | 277.1k     | 3.5m     | 1x15x13x48,4x2          | 1x17x15x48   | Type=padoptions                                        |
    | 14    | depthwise_conv_2d | 56.4k  | 24.2k  | 45.3k      | 82.0k      | 1.1m     | 1x17x15x48,1x3x3x48,48  | 1x8x7x48     | Multiplier:1 padding:valid stride:2x2 activation:relu6 |
    | 15    | conv_2d           | 43.5k  | 21.5k  | 18.9k      | 5.5k       | 270.0u   | 1x8x7x48,8x1x1x48,8     | 1x8x7x8      | Padding:same stride:1x1 activation:none                |
    | 16    | conv_2d           | 51.1k  | 21.5k  | 24.5k      | 5.3k       | 390.0u   | 1x8x7x8,48x1x1x8,48     | 1x8x7x48     | Padding:same stride:1x1 activation:relu6               |
    | 17    | depthwise_conv_2d | 56.4k  | 24.2k  | 38.8k      | 81.0k      | 1.0m     | 1x8x7x48,1x3x3x48,48    | 1x8x7x48     | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 18    | conv_2d           | 43.5k  | 21.5k  | 19.2k      | 5.4k       | 300.0u   | 1x8x7x48,8x1x1x48,8     | 1x8x7x8      | Padding:same stride:1x1 activation:none                |
    | 19    | add               | 448.0  | 0      | 1.6k       | 2.7k       | 60.0u    | 1x8x7x8,1x8x7x8         | 1x8x7x8      | Activation:none                                        |
    | 20    | conv_2d           | 51.1k  | 21.5k  | 24.5k      | 5.3k       | 360.0u   | 1x8x7x8,48x1x1x8,48     | 1x8x7x48     | Padding:same stride:1x1 activation:relu6               |
    | 21    | depthwise_conv_2d | 56.4k  | 24.2k  | 38.8k      | 81.1k      | 1.1m     | 1x8x7x48,1x3x3x48,48    | 1x8x7x48     | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 22    | conv_2d           | 43.5k  | 21.5k  | 19.2k      | 5.4k       | 270.0u   | 1x8x7x48,8x1x1x48,8     | 1x8x7x8      | Padding:same stride:1x1 activation:none                |
    | 23    | add               | 448.0  | 0      | 1.6k       | 2.7k       | 60.0u    | 1x8x7x8,1x8x7x8         | 1x8x7x8      | Activation:none                                        |
    | 24    | conv_2d           | 51.1k  | 21.5k  | 24.6k      | 5.3k       | 390.0u   | 1x8x7x8,48x1x1x8,48     | 1x8x7x48     | Padding:same stride:1x1 activation:relu6               |
    | 25    | pad               | 23.3k  | 0      | 0          | 81.4k      | 1.0m     | 1x8x7x48,4x2            | 1x9x9x48     | Type=padoptions                                        |
    | 26    | depthwise_conv_2d | 16.1k  | 6.9k   | 12.9k      | 24.3k      | 330.0u   | 1x9x9x48,1x3x3x48,48    | 1x4x4x48     | Multiplier:1 padding:valid stride:2x2 activation:relu6 |
    | 27    | conv_2d           | 24.8k  | 12.3k  | 10.6k      | 5.5k       | 180.0u   | 1x4x4x48,16x1x1x48,16   | 1x4x4x16     | Padding:same stride:1x1 activation:none                |
    | 28    | conv_2d           | 53.8k  | 24.6k  | 23.4k      | 5.3k       | 360.0u   | 1x4x4x16,96x1x1x16,96   | 1x4x4x96     | Padding:same stride:1x1 activation:relu6               |
    | 29    | depthwise_conv_2d | 32.2k  | 13.8k  | 18.9k      | 24.3k      | 390.0u   | 1x4x4x96,1x3x3x96,96    | 1x4x4x96     | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 30    | conv_2d           | 49.4k  | 24.6k  | 20.4k      | 5.4k       | 300.0u   | 1x4x4x96,16x1x1x96,16   | 1x4x4x16     | Padding:same stride:1x1 activation:none                |
    | 31    | add               | 256.0  | 0      | 908.0      | 2.7k       | 30.0u    | 1x4x4x16,1x4x4x16       | 1x4x4x16     | Activation:none                                        |
    | 32    | conv_2d           | 53.8k  | 24.6k  | 23.4k      | 5.3k       | 360.0u   | 1x4x4x16,96x1x1x16,96   | 1x4x4x96     | Padding:same stride:1x1 activation:relu6               |
    | 33    | depthwise_conv_2d | 32.2k  | 13.8k  | 18.9k      | 24.4k      | 390.0u   | 1x4x4x96,1x3x3x96,96    | 1x4x4x96     | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 34    | conv_2d           | 49.4k  | 24.6k  | 20.4k      | 5.4k       | 300.0u   | 1x4x4x96,16x1x1x96,16   | 1x4x4x16     | Padding:same stride:1x1 activation:none                |
    | 35    | add               | 256.0  | 0      | 908.0      | 2.7k       | 60.0u    | 1x4x4x16,1x4x4x16       | 1x4x4x16     | Activation:none                                        |
    | 36    | conv_2d           | 53.8k  | 24.6k  | 23.4k      | 5.3k       | 360.0u   | 1x4x4x16,96x1x1x16,96   | 1x4x4x96     | Padding:same stride:1x1 activation:relu6               |
    | 37    | depthwise_conv_2d | 32.2k  | 13.8k  | 18.9k      | 24.4k      | 390.0u   | 1x4x4x96,1x3x3x96,96    | 1x4x4x96     | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 38    | conv_2d           | 49.4k  | 24.6k  | 20.4k      | 5.4k       | 300.0u   | 1x4x4x96,16x1x1x96,16   | 1x4x4x16     | Padding:same stride:1x1 activation:none                |
    | 39    | add               | 256.0  | 0      | 908.0      | 2.7k       | 60.0u    | 1x4x4x16,1x4x4x16       | 1x4x4x16     | Activation:none                                        |
    | 40    | conv_2d           | 53.8k  | 24.6k  | 23.4k      | 5.3k       | 360.0u   | 1x4x4x16,96x1x1x16,96   | 1x4x4x96     | Padding:same stride:1x1 activation:relu6               |
    | 41    | depthwise_conv_2d | 32.2k  | 13.8k  | 18.9k      | 24.4k      | 390.0u   | 1x4x4x96,1x3x3x96,96    | 1x4x4x96     | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 42    | conv_2d           | 49.4k  | 24.6k  | 20.4k      | 5.4k       | 300.0u   | 1x4x4x96,16x1x1x96,16   | 1x4x4x16     | Padding:same stride:1x1 activation:none                |
    | 43    | add               | 256.0  | 0      | 908.0      | 2.7k       | 30.0u    | 1x4x4x16,1x4x4x16       | 1x4x4x16     | Activation:none                                        |
    | 44    | conv_2d           | 53.8k  | 24.6k  | 23.4k      | 5.3k       | 360.0u   | 1x4x4x16,96x1x1x16,96   | 1x4x4x96     | Padding:same stride:1x1 activation:relu6               |
    | 45    | depthwise_conv_2d | 32.2k  | 13.8k  | 18.9k      | 24.4k      | 390.0u   | 1x4x4x96,1x3x3x96,96    | 1x4x4x96     | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 46    | conv_2d           | 49.4k  | 24.6k  | 20.4k      | 5.4k       | 330.0u   | 1x4x4x96,16x1x1x96,16   | 1x4x4x16     | Padding:same stride:1x1 activation:none                |
    | 47    | add               | 256.0  | 0      | 908.0      | 2.7k       | 30.0u    | 1x4x4x16,1x4x4x16       | 1x4x4x16     | Activation:none                                        |
    | 48    | conv_2d           | 53.8k  | 24.6k  | 23.4k      | 5.3k       | 360.0u   | 1x4x4x16,96x1x1x16,96   | 1x4x4x96     | Padding:same stride:1x1 activation:relu6               |
    | 49    | depthwise_conv_2d | 32.2k  | 13.8k  | 18.9k      | 24.4k      | 390.0u   | 1x4x4x96,1x3x3x96,96    | 1x4x4x96     | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 50    | conv_2d           | 49.4k  | 24.6k  | 20.4k      | 5.4k       | 330.0u   | 1x4x4x96,16x1x1x96,16   | 1x4x4x16     | Padding:same stride:1x1 activation:none                |
    | 51    | add               | 256.0  | 0      | 908.0      | 2.7k       | 30.0u    | 1x4x4x16,1x4x4x16       | 1x4x4x16     | Activation:none                                        |
    | 52    | conv_2d           | 53.8k  | 24.6k  | 23.4k      | 5.3k       | 330.0u   | 1x4x4x16,96x1x1x16,96   | 1x4x4x96     | Padding:same stride:1x1 activation:relu6               |
    | 53    | pad               | 14.4k  | 0      | 0          | 41.5k      | 510.0u   | 1x4x4x96,4x2            | 1x5x5x96     | Type=padoptions                                        |
    | 54    | depthwise_conv_2d | 8.1k   | 3.5k   | 6.3k       | 7.0k       | 150.0u   | 1x5x5x96,1x3x3x96,96    | 1x2x2x96     | Multiplier:1 padding:valid stride:2x2 activation:relu6 |
    | 55    | conv_2d           | 18.5k  | 9.2k   | 7.6k       | 5.4k       | 150.0u   | 1x2x2x96,24x1x1x96,24   | 1x2x2x24     | Padding:same stride:1x1 activation:none                |
    | 56    | conv_2d           | 29.4k  | 13.8k  | 12.4k      | 5.3k       | 240.0u   | 1x2x2x24,144x1x1x24,144 | 1x2x2x144    | Padding:same stride:1x1 activation:relu6               |
    | 57    | depthwise_conv_2d | 12.1k  | 5.2k   | 5.0k       | 6.9k       | 120.0u   | 1x2x2x144,1x3x3x144,144 | 1x2x2x144    | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 58    | conv_2d           | 27.7k  | 13.8k  | 11.2k      | 5.4k       | 180.0u   | 1x2x2x144,24x1x1x144,24 | 1x2x2x24     | Padding:same stride:1x1 activation:none                |
    | 59    | add               | 96.0   | 0      | 348.0      | 2.7k       | 30.0u    | 1x2x2x24,1x2x2x24       | 1x2x2x24     | Activation:none                                        |
    | 60    | conv_2d           | 29.4k  | 13.8k  | 12.4k      | 5.3k       | 210.0u   | 1x2x2x24,144x1x1x24,144 | 1x2x2x144    | Padding:same stride:1x1 activation:relu6               |
    | 61    | depthwise_conv_2d | 12.1k  | 5.2k   | 5.0k       | 7.0k       | 120.0u   | 1x2x2x144,1x3x3x144,144 | 1x2x2x144    | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 62    | conv_2d           | 27.7k  | 13.8k  | 11.2k      | 5.4k       | 210.0u   | 1x2x2x144,24x1x1x144,24 | 1x2x2x24     | Padding:same stride:1x1 activation:none                |
    | 63    | add               | 96.0   | 0      | 348.0      | 2.7k       | 60.0u    | 1x2x2x24,1x2x2x24       | 1x2x2x24     | Activation:none                                        |
    | 64    | conv_2d           | 29.4k  | 13.8k  | 12.4k      | 5.3k       | 210.0u   | 1x2x2x24,144x1x1x24,144 | 1x2x2x144    | Padding:same stride:1x1 activation:relu6               |
    | 65    | depthwise_conv_2d | 12.1k  | 5.2k   | 5.0k       | 7.0k       | 120.0u   | 1x2x2x144,1x3x3x144,144 | 1x2x2x144    | Multiplier:1 padding:same stride:1x1 activation:relu6  |
    | 66    | conv_2d           | 55.5k  | 27.6k  | 21.8k      | 5.4k       | 330.0u   | 1x2x2x144,48x1x1x144,48 | 1x2x2x48     | Padding:same stride:1x1 activation:none                |
    | 67    | conv_2d           | 152.1k | 73.7k  | 60.5k      | 5.3k       | 810.0u   | 1x2x2x48,384x1x1x48,384 | 1x2x2x384    | Padding:valid stride:1x1 activation:relu6              |
    | 68    | mean              | 0      | 0      | 0          | 246.9k     | 3.1m     | 1x2x2x384,2             | 1x384        | Type=reduceroptions                                    |
    | 69    | fully_connected   | 8.4k   | 4.2k   | 6.4k       | 2.2k       | 120.0u   | 1x384,11x384,11         | 1x11         | Activation:none                                        |
    | 70    | softmax           | 55.0   | 0      | 0          | 9.0k       | 90.0u    | 1x11                    | 1x11         | Type=softmaxoptions                                    |
    +-------+-------------------+--------+--------+------------+------------+----------+-------------------------+--------------+--------------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell

   mltk view keyword_spotting_with_transfer_learning --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/keyword_spotting_with_transfer_learning.tflite.png" target="_blank">
            <img src="../../../../_images/models/keyword_spotting_with_transfer_learning.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>


Model Specification
---------------------

..  literalinclude:: ../../../../../../../mltk/models/siliconlabs/keyword_spotting_with_transfer_learning.py
    :language: python
    :lines: 455-

"""
# pylint: disable=redefined-outer-name

from typing import List, Tuple
# Import the Tensorflow packages
# required to build the model layout
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Import the MLTK model object
# and necessary mixins
# Later in this script we configure the various properties
from mltk.core import (
    load_mltk_model,
    MltkModel,
    TrainMixin,
    AudioDatasetMixin,
    EvaluateClassifierMixin
)

# Import the Google speech_commands dataset package
# This manages downloading and extracting the dataset
from mltk.datasets.audio.speech_commands import speech_commands_v2

# Import the ParallelAudioDataGenerator
# This has two main jobs:
# 1. Process the Google speech_commands dataset and apply random augmentations during training
# 2. Generate a spectrogram using the AudioFeatureGenerator from each augmented audio sample
#    and give the spectrogram to Tensorflow for model training
from mltk.core.preprocess.audio.parallel_generator import ParallelAudioDataGenerator, ParallelProcessParams
# Import the AudioFeatureGeneratorSettings which we'll configure
# and give to the ParallelAudioDataGenerator
from mltk.core.preprocess.audio.audio_feature_generator import AudioFeatureGeneratorSettings
from mltk.models.shared import MobileNetV2


# Define a custom model object with the following 'mixins':
# - TrainMixin        - Provides classifier model training operations and settings
# - AudioDatasetMixin - Provides audio data generation operations and settings
# - EvaluateClassifierMixin     - Provides classifier evaluation operations and settings
# @mltk_model # NOTE: This tag is required for this model be discoverable
class MyModel(
    MltkModel,
    TrainMixin,
    AudioDatasetMixin,
    EvaluateClassifierMixin
):
    pass

# Instantiate our custom model object
# The rest of this script simply configures the properties
# of our custom model object
my_model = MyModel()


#################################################
# General Settings

# For better tracking, the version should be incremented any time a non-trivial change is made
# NOTE: The version is optional and not used directly used by the MLTK
my_model.version = 1
# Provide a brief description about what this model models
# This description goes in the "description" field of the .tflite model file
my_model.description = 'Keyword spotting classifier using transfer learning to detect: "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"'

#################################################
# Training Basic Settings

# This specifies the number of times we run the training
# samples through the model to update the model weights.
# Typically, a larger value leads to better accuracy at the expense of training time.
# Set to -1 to use the early_stopping callback and let the scripts
# determine how many epochs to train for (see below).
# Otherwise set this to a specific value (typically 40-200)
my_model.epochs = 10
# Specify how many samples to pass through the model
# before updating the training gradients.
# Typical values are 10-64
# NOTE: Larger values require more memory and may not fit on your GPU
my_model.batch_size = 16
# This specifies the algorithm used to update the model gradients
# during training. Adam is very common
# See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
my_model.optimizer = 'adam'
# List of metrics to be evaluated by the model during training and testing
my_model.metrics = ['accuracy']
# The "loss" function used to update the weights
# This is a classification problem with more than two labels so we use categorical_crossentropy
# See https://www.tensorflow.org/api_docs/python/tf/keras/losses
my_model.loss = 'categorical_crossentropy'


#################################################
# Training callback Settings

# Generate checkpoints every time the validation accuracy improves
# See https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
my_model.checkpoint['monitor'] =  'val_accuracy'

# https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler
# Update the learning rate each epoch based on the given callback
def lr_schedule(epoch):
    # When using transfer learning, the initial learning rate should start at a fairly small value
    initial_learning_rate = 0.0005
    decay_per_epoch = 0.95
    lrate = initial_learning_rate * (decay_per_epoch ** epoch)
    return lrate

my_model.lr_schedule = dict(
    schedule = lr_schedule,
    verbose = 1
)


#################################################
# TF-Lite converter settings

# These are the settings used to quantize the model
# We want all the internal ops as well as
# model input/output to be int8
my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# NOTE: A float32 model input/output is also possible
my_model.tflite_converter['inference_input_type'] = np.int8
my_model.tflite_converter['inference_output_type'] = np.int8
# Automatically generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'



#################################################
# Audio Data Provider Settings

# Specify the dataset
# NOTE: This can also be an absolute path to a directory
#       or a Python function
# See: https://siliconlabs.github.io/mltk/docs/python_api/mltk_model/audio_dataset_mixin.html#mltk.core.AudioDatasetMixin.dataset
my_model.dataset = speech_commands_v2
# We're using a 'categorical_crossentropy' loss
# so must also use a `categorical` class mode for the data generation
my_model.class_mode = 'categorical'

# Specify the keywords we want to detect
# In this model, we detect 'one', 'two', 'three', 'four',
# plus two pseudo classes: _unknown_ and _silence_
#
# NOTE: The more classes that are added, the larger the training dataset which cna greatly increase
#       the training time. So, if you want to quickly train a model then it is recommend to reduce the class count.
my_model.classes = ['one', 'two', 'three', 'four','five','six', 'seven','eight', 'nine', '_unknown_', '_silence_']

# Assign the weights used by each class while training
# A larger value makes the model focus more on the class
my_model.class_weights = 'balanced'



#################################################
# AudioFeatureGenerator Settings
#
# These are the settings used by the AudioFeatureGenerator
# to generate spectrograms from the audio samples
# These settings must be used during modeling training
# AND by embedded device at runtime
#
# See https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
frontend_settings = AudioFeatureGeneratorSettings()

frontend_settings.sample_rate_hz = 16000  # This can also be 16k for slightly better performance at the cost of more RAM
frontend_settings.sample_length_ms = 1200
frontend_settings.window_size_ms = 30 # Best to key this a power of 2 to aid the FFT
frontend_settings.window_step_ms = 20
frontend_settings.filterbank_n_channels = 49
frontend_settings.filterbank_upper_band_limit = 4000.0-1 # Spoken language usually only goes up to 4k
frontend_settings.filterbank_lower_band_limit = 125.0
frontend_settings.noise_reduction_enable = True
frontend_settings.noise_reduction_smoothing_bits = 10
frontend_settings.noise_reduction_even_smoothing =  0.025
frontend_settings.noise_reduction_odd_smoothing = 0.06
frontend_settings.noise_reduction_min_signal_remaining = 0.03
frontend_settings.pcan_enable = False
frontend_settings.pcan_strength = 0.95
frontend_settings.pcan_offset = 80.0
frontend_settings.pcan_gain_bits = 21
frontend_settings.log_scale_enable = True
frontend_settings.log_scale_shift = 6




#################################################
# ParallelAudioDataGenerator Settings
#
# Configure the data generator settings
# This specifies how to augment the training samples
# See the command: "mltk view_audio"
# to get a better idea of how these augmentations affect
# the samples
my_model.datagen = ParallelAudioDataGenerator(
    dtype=my_model.tflite_converter['inference_input_type'],
    frontend_settings=frontend_settings,
    cores=0.65, # Adjust this as necessary for your PC setup
    debug=False, # Set this to true to enable debugging of the generator
    # debug=True,
    # cores=1,
    max_batches_pending=32,  # Adjust this as necessary for your PC setup (smaller -> less RAM)
    validation_split= 0.15,
    validation_augmentation_enabled=False,
    samplewise_center=False,
    samplewise_std_normalization=False,
    rescale=None,
    unknown_class_percentage=1.0, # Increasing this may help model robustness at the expense of training time
    silence_class_percentage=0.2,
    offset_range=(0.0,1.0),
    trim_threshold_db=20,
    noise_colors=None,
    #loudness_range=(0.5, 1.0),
    # speed_range=(0.9,1.1),
    # pitch_range=(0.9,1.1),
    # vtlp_range=(0.9,1.1),
    bg_noise_range=(0.0,0.4),
    bg_noise_dir='_background_noise_', # This is a directory provided by the google speech commands dataset, can also provide an absolute path
)




#################################################
# Model Layout
#
# This defines the actual model layout
# using the Keras API.
def my_model_builder(model: MyModel):
    # Create an instance of the MobileNetV2
    # NOTE: This should have similar parameters to the keyword_spotting_mobilenetv2 model
    #       since we're transferring weights from it
    keras_model = MobileNetV2(
        input_shape=model.input_shape,
        classes=model.n_classes,
        alpha=0.15,
        last_block_filters=384,
        include_top=True,
        weights=None
    )

    # Load the "keyword_spotting_mobilenetv2" model
    # We want to transfer its weights to this model
    # In this way, this new model can start with the knowledge
    # that the keyword_spotting_mobilenetv2 model already knows
    # NOTE: This step is not needed if you already have a .h5 file
    base_mltk_model = load_mltk_model('keyword_spotting_mobilenetv2')

    # Get the file path to the .h5 file found in the keyword_spotting_mobilenetv2 model archive
    # The .h5 file contains the trained weights we want to transfer to this model
    base_model_h5_path = base_mltk_model.h5_archive_path

    # Load the keyword_spotting_mobilenetv2 weights into this model
    keras_model.load_weights(
        base_model_h5_path,
        by_name=True,
        skip_mismatch=True # We need to skip mismatches in case the number of classes is different
    )

    # NOTE: The https://keras.io/guides/transfer_learning recommends
    #       "freezing" layers of the base model during training, however, in this instance,
    #       it was found that making all layers trainable gave better performance.

    keras_model.compile(
        loss=model.loss,
        optimizer=model.optimizer,
        metrics=model.metrics
    )
    return keras_model

my_model.build_model_function = my_model_builder



#################################################
# Audio Classifier Settings
#
# These are additional parameters to include in
# the generated .tflite model file.
# The settings are used by the "classify_audio" command
# or audio_classifier example application.
# NOTE: Corresponding command-line options will override these values.


# Controls the smoothing.
# Drop all inference results that are older than <now> minus window_duration
# Longer durations (in milliseconds) will give a higher confidence that the results are correct, but may miss some commands
my_model.model_parameters['average_window_duration_ms'] = 1000

# Minimum averaged model output threshold for a class to be considered detected, 0-255. Higher values increase precision at the cost of recall
my_model.model_parameters['detection_threshold'] = 145

# Amount of milliseconds to wait after a keyword is detected before detecting new keywords
my_model.model_parameters['suppression_ms'] = 750

# The minimum number of inference results to average when calculating the detection value
my_model.model_parameters['minimum_count'] = 3

# Set the volume gain scaler (i.e. amplitude) to apply to the microphone data. If 0 or omitted, no scaler is applied
my_model.model_parameters['volume_gain'] = 2

# This the amount of time in milliseconds an audio loop takes.
my_model.model_parameters['latency_ms'] = 100

# Enable verbose inference results
my_model.model_parameters['verbose_model_output_logs'] = False




##########################################################################################
# The following allows for running this model training script directly, e.g.:
# python keyword_spotting_with_transfer_learning.py
#
# Note that this has the same functionality as:
# mltk train keyword_spotting_with_transfer_learning
#
if __name__ == '__main__':
    import mltk.core as mltk_core
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=False)

    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command: mltk train keyword_spotting_with_transfer_learning-test --clean
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate keyword_spotting_with_transfer_learning-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile keyword_spotting_with_transfer_learning-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)