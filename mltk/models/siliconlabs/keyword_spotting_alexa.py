"""keyword_spotting_alexa
*******************************

- Source code: `keyword_spotting_alexa.py <https://github.com/siliconlabs/mltk/blob/master/mltk/models/siliconlabs/keyword_spotting_alexa.py>`_
- Pre-trained model: `keyword_spotting_alexa.mltk.zip <https://github.com/SiliconLabs/mltk/raw/master/mltk/models/siliconlabs/keyword_spotting_alexa.mltk.zip>`_

This model is designed to detect the keyword: "Alexa".

It based on the `Temporal Efficient Neural Network (TENet) <https://arxiv.org/pdf/2010.09960.pdf>`_ model architecture:

   This is a keyword spotting architecture with temporal and depthwise convolutions.


This model specification script is designed to work with the
`Keyword Spotting Alexa <https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_alexa.html>`_ tutorial.


Dataset
---------

This combines several different datasets:

- A `synthetically generated <https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html>`_ "Alexa" dataset - Different computer-generated audio clips of the keyword "alexa"
- A `synthetically generated <https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html>`_ "unknown" class - Different computer-generated audio clips that sound similar to "alexa"; used for the "unknown" class; helps avoid false-positives
- A subset of the `MLCommons Multilingual Spoken Words <https://mlcommons.org/en/multilingual-spoken-words>`_ dataset - Used for the "unknown" class; helps to avoid false-positives
- A subset of the `Mozilla Common Voice <https://commonvoice.mozilla.org/en/datasets>`_  dataset - Used for the "unknown" class; helps to avoid false-positives



Preprocessing
--------------
This uses the :py:class:`mltk.core.preprocess.audio.audio_feature_generator.AudioFeatureGenerator` to generate spectrograms with the settings:

- sample_rate: 16kHz
- sample_length: 1200ms
- window size: 30ms
- window step: 10ms
- n_channels: 108
- noise_reduction_enable: 1
- noise_reduction_min_signal_remaining: 0.40


Commands
--------------

.. code-block:: shell

   # Do a "dry run" test training of the model
   mltk train keyword_spotting_alexa-test

   # Train the model
   mltk train keyword_spotting_alexa

   # Evaluate the trained model .tflite model
   mltk evaluate keyword_spotting_alexa --tflite

   # Profile the model in the MVP hardware accelerator simulator
   mltk profile keyword_spotting_alexa --accelerator MVP

   # Profile the model on a physical development board
   mltk profile keyword_spotting_alexa  --accelerator MVP --device

   # Run the model in the audio classifier on the local PC
   mltk classify_audio keyword_spotting_alexa --verbose

   # Run the model in the audio classifier on the physical device feature an MVP hardware accelerator
   mltk classify_audio keyword_spotting_alexa --device  --accelerator MVP --verbose


Model Summary
--------------

.. code-block:: shell

    mltk summarize keyword_spotting_alexa --tflite

    +-------+-------------------+------------------+-----------------+-------------------------------------------------------+
    | Index | OpCode            | Input(s)         | Output(s)       | Config                                                |
    +-------+-------------------+------------------+-----------------+-------------------------------------------------------+
    | 0     | conv_2d           | 118x1x108 (int8) | 118x1x32 (int8) | Padding:Same stride:1x1 activation:None               |
    |       |                   | 3x1x108 (int8)   |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 1     | conv_2d           | 118x1x32 (int8)  | 118x1x96 (int8) | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 2     | depthwise_conv_2d | 118x1x96 (int8)  | 59x1x96 (int8)  | Multiplier:1 padding:Same stride:2x2 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 3     | conv_2d           | 59x1x96 (int8)   | 59x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 4     | conv_2d           | 118x1x32 (int8)  | 59x1x32 (int8)  | Padding:Same stride:2x2 activation:Relu               |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 5     | add               | 59x1x32 (int8)   | 59x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 59x1x32 (int8)   |                 |                                                       |
    | 6     | conv_2d           | 59x1x32 (int8)   | 59x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 7     | depthwise_conv_2d | 59x1x96 (int8)   | 59x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 8     | conv_2d           | 59x1x96 (int8)   | 59x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 9     | add               | 59x1x32 (int8)   | 59x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 59x1x32 (int8)   |                 |                                                       |
    | 10    | conv_2d           | 59x1x32 (int8)   | 59x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 11    | depthwise_conv_2d | 59x1x96 (int8)   | 59x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 12    | conv_2d           | 59x1x96 (int8)   | 59x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 13    | add               | 59x1x32 (int8)   | 59x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 59x1x32 (int8)   |                 |                                                       |
    | 14    | conv_2d           | 59x1x32 (int8)   | 59x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 15    | depthwise_conv_2d | 59x1x96 (int8)   | 59x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 16    | conv_2d           | 59x1x96 (int8)   | 59x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 17    | add               | 59x1x32 (int8)   | 59x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 59x1x32 (int8)   |                 |                                                       |
    | 18    | conv_2d           | 59x1x32 (int8)   | 59x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 19    | depthwise_conv_2d | 59x1x96 (int8)   | 30x1x96 (int8)  | Multiplier:1 padding:Same stride:2x2 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 20    | conv_2d           | 30x1x96 (int8)   | 30x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 21    | conv_2d           | 59x1x32 (int8)   | 30x1x32 (int8)  | Padding:Same stride:2x2 activation:Relu               |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 22    | add               | 30x1x32 (int8)   | 30x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 30x1x32 (int8)   |                 |                                                       |
    | 23    | conv_2d           | 30x1x32 (int8)   | 30x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 24    | depthwise_conv_2d | 30x1x96 (int8)   | 30x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 25    | conv_2d           | 30x1x96 (int8)   | 30x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 26    | add               | 30x1x32 (int8)   | 30x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 30x1x32 (int8)   |                 |                                                       |
    | 27    | conv_2d           | 30x1x32 (int8)   | 30x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 28    | depthwise_conv_2d | 30x1x96 (int8)   | 30x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 29    | conv_2d           | 30x1x96 (int8)   | 30x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 30    | add               | 30x1x32 (int8)   | 30x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 30x1x32 (int8)   |                 |                                                       |
    | 31    | conv_2d           | 30x1x32 (int8)   | 30x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 32    | depthwise_conv_2d | 30x1x96 (int8)   | 30x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 33    | conv_2d           | 30x1x96 (int8)   | 30x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 34    | add               | 30x1x32 (int8)   | 30x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 30x1x32 (int8)   |                 |                                                       |
    | 35    | conv_2d           | 30x1x32 (int8)   | 30x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 36    | depthwise_conv_2d | 30x1x96 (int8)   | 15x1x96 (int8)  | Multiplier:1 padding:Same stride:2x2 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 37    | conv_2d           | 15x1x96 (int8)   | 15x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 38    | conv_2d           | 30x1x32 (int8)   | 15x1x32 (int8)  | Padding:Same stride:2x2 activation:Relu               |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 39    | add               | 15x1x32 (int8)   | 15x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 15x1x32 (int8)   |                 |                                                       |
    | 40    | conv_2d           | 15x1x32 (int8)   | 15x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 41    | depthwise_conv_2d | 15x1x96 (int8)   | 15x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 42    | conv_2d           | 15x1x96 (int8)   | 15x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 43    | add               | 15x1x32 (int8)   | 15x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 15x1x32 (int8)   |                 |                                                       |
    | 44    | conv_2d           | 15x1x32 (int8)   | 15x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 45    | depthwise_conv_2d | 15x1x96 (int8)   | 15x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 46    | conv_2d           | 15x1x96 (int8)   | 15x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 47    | add               | 15x1x32 (int8)   | 15x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 15x1x32 (int8)   |                 |                                                       |
    | 48    | conv_2d           | 15x1x32 (int8)   | 15x1x96 (int8)  | Padding:Valid stride:1x1 activation:Relu              |
    |       |                   | 1x1x32 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 49    | depthwise_conv_2d | 15x1x96 (int8)   | 15x1x96 (int8)  | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    |       |                   | 9x1x96 (int8)    |                 |                                                       |
    |       |                   | 96 (int32)       |                 |                                                       |
    | 50    | conv_2d           | 15x1x96 (int8)   | 15x1x32 (int8)  | Padding:Valid stride:1x1 activation:None              |
    |       |                   | 1x1x96 (int8)    |                 |                                                       |
    |       |                   | 32 (int32)       |                 |                                                       |
    | 51    | add               | 15x1x32 (int8)   | 15x1x32 (int8)  | Activation:Relu                                       |
    |       |                   | 15x1x32 (int8)   |                 |                                                       |
    | 52    | average_pool_2d   | 15x1x32 (int8)   | 1x1x32 (int8)   | Padding:Valid stride:1x15 filter:1x15 activation:None |
    | 53    | reshape           | 1x1x32 (int8)    | 32 (int8)       | Type=none                                             |
    |       |                   | 2 (int32)        |                 |                                                       |
    | 54    | fully_connected   | 32 (int8)        | 2 (int8)        | Activation:None                                       |
    |       |                   | 32 (int8)        |                 |                                                       |
    |       |                   | 2 (int32)        |                 |                                                       |
    | 55    | softmax           | 2 (int8)         | 2 (int8)        | Type=softmaxoptions                                   |
    +-------+-------------------+------------------+-----------------+-------------------------------------------------------+
    Total MACs: 4.562 M
    Total OPs: 9.247 M
    Name: keyword_spotting_alexa_v2
    Version: 2
    Description: Keyword spotting classifier to detect: "alexa"
    Classes: alexa, _unknown_
    Runtime memory size (RAM): 54.344 k
    hash: 026c2f86bf499c3a1386c348888021e5
    date: 2022-12-10T00:29:35.325Z
    fe.sample_rate_hz: 16000
    fe.fft_length: 512
    fe.sample_length_ms: 1200
    fe.window_size_ms: 30
    fe.window_step_ms: 10
    fe.filterbank_n_channels: 108
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
    fe.quantize_dynamic_scale_enable: True
    fe.quantize_dynamic_scale_range_db: 40.0
    latency_ms: 200
    minimum_count: 2
    average_window_duration_ms: 440
    detection_threshold: 216
    suppression_ms: 900
    volume_gain: 0
    verbose_model_output_logs: True
    .tflite file size: 208.1kB


Model Profiling Report
-----------------------

.. code-block:: shell

   # Profile on physical EFR32xG24 using MVP accelerator
   mltk profile keyword_spotting_alexa --device --accelerator MVP

    Profiling Summary
    Name: keyword_spotting_alexa
    Accelerator: MVP
    Input Shape: 1x118x1x108
    Input Data Type: int8
    Output Shape: 1x2
    Output Data Type: int8
    Flash, Model File Size (bytes): 207.4k
    RAM, Runtime Memory Size (bytes): 65.1k
    Operation Count: 9.4M
    Multiply-Accumulate Count: 4.6M
    Layer Count: 56
    Unsupported Layer Count: 0
    Accelerator Cycle Count: 4.1M
    CPU Cycle Count: 825.0k
    CPU Utilization (%): 18.6
    Clock Rate (hz): 78.0M
    Time (s): 57.0m
    Ops/s: 165.5M
    MACs/s: 80.0M
    Inference/s: 17.5

    Model Layers
    +-------+-------------------+--------+--------+------------+------------+----------+---------------------------+--------------+-------------------------------------------------------+
    | Index | OpCode            | # Ops  | # MACs | Acc Cycles | CPU Cycles | Time (s) | Input Shape               | Output Shape | Options                                               |
    +-------+-------------------+--------+--------+------------+------------+----------+---------------------------+--------------+-------------------------------------------------------+
    | 0     | conv_2d           | 2.5M   | 1.2M   | 930.7k     | 11.3k      | 11.8m    | 1x118x1x108,32x3x1x108,32 | 1x118x1x32   | Padding:Same stride:1x1 activation:None               |
    | 1     | conv_2d           | 759.0k | 362.5k | 307.9k     | 5.2k       | 3.9m     | 1x118x1x32,96x1x1x32,96   | 1x118x1x96   | Padding:Valid stride:1x1 activation:Relu              |
    | 2     | depthwise_conv_2d | 118.9k | 51.0k  | 91.9k      | 88.9k      | 1.6m     | 1x118x1x96,1x9x1x96,96    | 1x59x1x96    | Multiplier:1 padding:Same stride:2x2 activation:Relu  |
    | 3     | conv_2d           | 364.4k | 181.2k | 145.7k     | 5.3k       | 1.9m     | 1x59x1x96,32x1x1x96,32    | 1x59x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 4     | conv_2d           | 126.5k | 60.4k  | 52.9k      | 5.1k       | 690.0u   | 1x118x1x32,32x1x1x32,32   | 1x59x1x32    | Padding:Same stride:2x2 activation:Relu               |
    | 5     | add               | 1.9k   | 0      | 6.6k       | 2.8k       | 90.0u    | 1x59x1x32,1x59x1x32       | 1x59x1x32    | Activation:Relu                                       |
    | 6     | conv_2d           | 379.5k | 181.2k | 154.0k     | 5.2k       | 2.0m     | 1x59x1x32,96x1x1x32,96    | 1x59x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 7     | depthwise_conv_2d | 118.9k | 51.0k  | 90.4k      | 88.7k      | 1.6m     | 1x59x1x96,1x9x1x96,96     | 1x59x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 8     | conv_2d           | 364.4k | 181.2k | 145.7k     | 5.3k       | 1.9m     | 1x59x1x96,32x1x1x96,32    | 1x59x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 9     | add               | 1.9k   | 0      | 6.6k       | 2.7k       | 120.0u   | 1x59x1x32,1x59x1x32       | 1x59x1x32    | Activation:Relu                                       |
    | 10    | conv_2d           | 379.5k | 181.2k | 154.0k     | 5.2k       | 2.0m     | 1x59x1x32,96x1x1x32,96    | 1x59x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 11    | depthwise_conv_2d | 118.9k | 51.0k  | 90.4k      | 88.7k      | 1.6m     | 1x59x1x96,1x9x1x96,96     | 1x59x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 12    | conv_2d           | 364.4k | 181.2k | 145.7k     | 5.3k       | 1.9m     | 1x59x1x96,32x1x1x96,32    | 1x59x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 13    | add               | 1.9k   | 0      | 6.6k       | 2.7k       | 120.0u   | 1x59x1x32,1x59x1x32       | 1x59x1x32    | Activation:Relu                                       |
    | 14    | conv_2d           | 379.5k | 181.2k | 154.0k     | 5.2k       | 2.0m     | 1x59x1x32,96x1x1x32,96    | 1x59x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 15    | depthwise_conv_2d | 118.9k | 51.0k  | 90.4k      | 88.7k      | 1.6m     | 1x59x1x96,1x9x1x96,96     | 1x59x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 16    | conv_2d           | 364.4k | 181.2k | 145.7k     | 5.3k       | 1.9m     | 1x59x1x96,32x1x1x96,32    | 1x59x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 17    | add               | 1.9k   | 0      | 6.6k       | 2.7k       | 120.0u   | 1x59x1x32,1x59x1x32       | 1x59x1x32    | Activation:Relu                                       |
    | 18    | conv_2d           | 379.5k | 181.2k | 154.5k     | 5.2k       | 2.0m     | 1x59x1x32,96x1x1x32,96    | 1x59x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 19    | depthwise_conv_2d | 60.5k  | 25.9k  | 45.7k      | 45.6k      | 840.0u   | 1x59x1x96,1x9x1x96,96     | 1x30x1x96    | Multiplier:1 padding:Same stride:2x2 activation:Relu  |
    | 20    | conv_2d           | 185.3k | 92.2k  | 74.2k      | 5.3k       | 960.0u   | 1x30x1x96,32x1x1x96,32    | 1x30x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 21    | conv_2d           | 64.3k  | 30.7k  | 27.4k      | 5.1k       | 390.0u   | 1x59x1x32,32x1x1x32,32    | 1x30x1x32    | Padding:Same stride:2x2 activation:Relu               |
    | 22    | add               | 960.0  | 0      | 3.4k       | 2.7k       | 90.0u    | 1x30x1x32,1x30x1x32       | 1x30x1x32    | Activation:Relu                                       |
    | 23    | conv_2d           | 193.0k | 92.2k  | 78.6k      | 5.2k       | 1.1m     | 1x30x1x32,96x1x1x32,96    | 1x30x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 24    | depthwise_conv_2d | 60.5k  | 25.9k  | 44.6k      | 45.7k      | 840.0u   | 1x30x1x96,1x9x1x96,96     | 1x30x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 25    | conv_2d           | 185.3k | 92.2k  | 74.2k      | 5.3k       | 960.0u   | 1x30x1x96,32x1x1x96,32    | 1x30x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 26    | add               | 960.0  | 0      | 3.4k       | 2.7k       | 90.0u    | 1x30x1x32,1x30x1x32       | 1x30x1x32    | Activation:Relu                                       |
    | 27    | conv_2d           | 193.0k | 92.2k  | 78.6k      | 5.2k       | 1.1m     | 1x30x1x32,96x1x1x32,96    | 1x30x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 28    | depthwise_conv_2d | 60.5k  | 25.9k  | 44.6k      | 45.7k      | 810.0u   | 1x30x1x96,1x9x1x96,96     | 1x30x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 29    | conv_2d           | 185.3k | 92.2k  | 74.2k      | 5.3k       | 960.0u   | 1x30x1x96,32x1x1x96,32    | 1x30x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 30    | add               | 960.0  | 0      | 3.4k       | 2.7k       | 90.0u    | 1x30x1x32,1x30x1x32       | 1x30x1x32    | Activation:Relu                                       |
    | 31    | conv_2d           | 193.0k | 92.2k  | 78.6k      | 5.2k       | 1.1m     | 1x30x1x32,96x1x1x32,96    | 1x30x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 32    | depthwise_conv_2d | 60.5k  | 25.9k  | 44.6k      | 45.7k      | 810.0u   | 1x30x1x96,1x9x1x96,96     | 1x30x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 33    | conv_2d           | 185.3k | 92.2k  | 74.2k      | 5.3k       | 990.0u   | 1x30x1x96,32x1x1x96,32    | 1x30x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 34    | add               | 960.0  | 0      | 3.4k       | 2.7k       | 90.0u    | 1x30x1x32,1x30x1x32       | 1x30x1x32    | Activation:Relu                                       |
    | 35    | conv_2d           | 193.0k | 92.2k  | 78.6k      | 5.2k       | 1.1m     | 1x30x1x32,96x1x1x32,96    | 1x30x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 36    | depthwise_conv_2d | 30.2k  | 13.0k  | 22.3k      | 23.4k      | 420.0u   | 1x30x1x96,1x9x1x96,96     | 1x15x1x96    | Multiplier:1 padding:Same stride:2x2 activation:Relu  |
    | 37    | conv_2d           | 92.6k  | 46.1k  | 37.2k      | 5.2k       | 510.0u   | 1x15x1x96,32x1x1x96,32    | 1x15x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 38    | conv_2d           | 32.2k  | 15.4k  | 13.7k      | 5.1k       | 240.0u   | 1x30x1x32,32x1x1x32,32    | 1x15x1x32    | Padding:Same stride:2x2 activation:Relu               |
    | 39    | add               | 480.0  | 0      | 1.7k       | 2.7k       | 60.0u    | 1x15x1x32,1x15x1x32       | 1x15x1x32    | Activation:Relu                                       |
    | 40    | conv_2d           | 96.5k  | 46.1k  | 39.4k      | 5.2k       | 570.0u   | 1x15x1x32,96x1x1x32,96    | 1x15x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 41    | depthwise_conv_2d | 30.2k  | 13.0k  | 20.8k      | 23.4k      | 390.0u   | 1x15x1x96,1x9x1x96,96     | 1x15x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 42    | conv_2d           | 92.6k  | 46.1k  | 37.2k      | 5.2k       | 540.0u   | 1x15x1x96,32x1x1x96,32    | 1x15x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 43    | add               | 480.0  | 0      | 1.7k       | 2.7k       | 60.0u    | 1x15x1x32,1x15x1x32       | 1x15x1x32    | Activation:Relu                                       |
    | 44    | conv_2d           | 96.5k  | 46.1k  | 39.4k      | 5.2k       | 540.0u   | 1x15x1x32,96x1x1x32,96    | 1x15x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 45    | depthwise_conv_2d | 30.2k  | 13.0k  | 20.8k      | 23.4k      | 420.0u   | 1x15x1x96,1x9x1x96,96     | 1x15x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 46    | conv_2d           | 92.6k  | 46.1k  | 37.2k      | 5.2k       | 510.0u   | 1x15x1x96,32x1x1x96,32    | 1x15x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 47    | add               | 480.0  | 0      | 1.7k       | 2.7k       | 30.0u    | 1x15x1x32,1x15x1x32       | 1x15x1x32    | Activation:Relu                                       |
    | 48    | conv_2d           | 96.5k  | 46.1k  | 39.4k      | 5.2k       | 570.0u   | 1x15x1x32,96x1x1x32,96    | 1x15x1x96    | Padding:Valid stride:1x1 activation:Relu              |
    | 49    | depthwise_conv_2d | 30.2k  | 13.0k  | 20.8k      | 23.4k      | 420.0u   | 1x15x1x96,1x9x1x96,96     | 1x15x1x96    | Multiplier:1 padding:Same stride:1x1 activation:Relu  |
    | 50    | conv_2d           | 92.6k  | 46.1k  | 37.2k      | 5.2k       | 510.0u   | 1x15x1x96,32x1x1x96,32    | 1x15x1x32    | Padding:Valid stride:1x1 activation:None              |
    | 51    | add               | 480.0  | 0      | 1.7k       | 2.7k       | 60.0u    | 1x15x1x32,1x15x1x32       | 1x15x1x32    | Activation:Relu                                       |
    | 52    | average_pool_2d   | 512.0  | 0      | 309.0      | 3.9k       | 60.0u    | 1x15x1x32                 | 1x1x1x32     | Padding:Valid stride:1x15 filter:1x15 activation:None |
    | 53    | reshape           | 0      | 0      | 0          | 595.0      | 0        | 1x1x1x32,2                | 1x32         | Type=none                                             |
    | 54    | fully_connected   | 130.0  | 64.0   | 123.0      | 2.1k       | 30.0u    | 1x32,2x32,2               | 1x2          | Activation:None                                       |
    | 55    | softmax           | 10.0   | 0      | 0          | 2.4k       | 30.0u    | 1x2                       | 1x2          | Type=softmaxoptions                                   |
    +-------+-------------------+--------+--------+------------+------------+----------+---------------------------+--------------+-------------------------------------------------------+


Model Diagram
------------------

.. code-block:: shell

   mltk view keyword_spotting_alexa --tflite

.. raw:: html

    <div class="model-diagram">
        <a href="../../../../_images/models/keyword_spotting_alexa.tflite.png" target="_blank">
            <img src="../../../../_images/models/keyword_spotting_alexa.tflite.png" />
            <p>Click to enlarge</p>
        </a>
    </div>



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
from mltk.utils.archive_downloader import download_verify_extract, download_url
from mltk.models.shared import tenet



##########################################################################################
# Instantiate the MltkModel instance
#

# @mltk_model
class MyModel(
    mltk_core.MltkModel,    # We must inherit the MltkModel class
    mltk_core.TrainMixin,   # We also inherit the TrainMixin since we want to train this model
    mltk_core.DatasetMixin, # We also need the DatasetMixin mixin to provide the relevant dataset properties
    mltk_core.EvaluateClassifierMixin,  # While not required, also inherit EvaluateClassifierMixin to help will generating evaluation stats for our classification model
    mltk_core.SshMixin,
):
    pass
my_model = MyModel()

##########################################################################################
# General Settings

# For better tracking, the version should be incremented any time a non-trivial change is made
# NOTE: The version is optional and not used directly used by the MLTK
my_model.version = 1
# Provide a brief description about what this model models
# This description goes in the "description" field of the .tflite model file
my_model.description = 'Keyword spotting classifier to detect: "alexa"'


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
my_model.batch_size = 100


##########################################################################################
# Define the model architecture
#

def my_model_builder(model: MyModel) -> tf.keras.Model:
    """Build the "Teacher" Keras model
    """
    input_shape = model.input_shape
    # NOTE: This model requires the input shape: <time, 1, features>
    #       while the embedded device expects: <time, features, 1>
    #       Since the <time> axis is still row-major, we can swap the <features> with 1 without issue
    time_size, feature_size, _ = input_shape
    input_shape = (time_size, 1, feature_size)

    keras_model = tenet.TENet12(
        input_shape=input_shape,
        classes=model.n_classes
    )

    keras_model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8),
        metrics= ['accuracy']
    )

    return keras_model

my_model.build_model_function = my_model_builder
my_model.keras_custom_objects['MultiScaleTemporalConvolution'] = tenet.MultiScaleTemporalConvolution



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
        (100,   .001),
        (100,   .002),
        (100,   .003),
        (100,   .004),
        (30000, .005),
        (30000, .002),
        (20000, .0005),
        (10000, 1e-5),
        (5000,  1e-6),
        (5000,  1e-7),
    ] )
]


##########################################################################################
# Specify AudioFeatureGenerator Settings
# See https://siliconlabs.github.io/mltk/docs/audio/audio_feature_generator.html
#
frontend_settings = AudioFeatureGeneratorSettings()

frontend_settings.sample_rate_hz = 16000
frontend_settings.sample_length_ms = 1200                       # Use 1.2s audio clips to ensure the full "alexa" keyword is captured
frontend_settings.window_size_ms = 30
frontend_settings.window_step_ms = 10
frontend_settings.filterbank_n_channels = 108                   # We want this value to be as large as possible
                                                                # while still allowing for the ML model to execute efficiently on the hardware
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

frontend_settings.quantize_dynamic_scale_enable = True          # Enable dynamic quantization, this dynamically converts the uint16 spectrogram to int8
frontend_settings.quantize_dynamic_scale_range_db = 40.0


# Add the Audio Feature generator settings to the model parameters
# This way, they are included in the generated .tflite model file
# See https://siliconlabs.github.io/mltk/docs/guides/model_parameters.html
my_model.model_parameters.update(frontend_settings)


##########################################################################################
# Specify the other dataset settings
#

my_model.input_shape = frontend_settings.spectrogram_shape + (1,)

# Add the direction keywords plus a _unknown_ meta class
my_model.classes = ['alexa', '_unknown_']
unknown_class_id = my_model.classes.index('_unknown_')

# Ensure the class weights are balanced during training
# https://towardsdatascience.com/why-weight-the-importance-of-training-on-balanced-datasets-f1e54688e7df
my_model.class_weights = 'balanced'


##########################################################################################
# TF-Lite converter settings
#

my_model.tflite_converter['optimizations'] = [tf.lite.Optimize.DEFAULT]
my_model.tflite_converter['supported_ops'] = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
my_model.tflite_converter['inference_input_type'] = np.int8
my_model.tflite_converter['inference_output_type'] = np.int8
# Automatically generate a representative dataset from the validation data
my_model.tflite_converter['representative_dataset'] = 'generate'


validation_split = 0.10
unknown_class_multiplier = 1.5 # This controls how many more "unknown" samples there are relative to the "known" samples

# Uncomment this to dump the augmented audio samples to the log directory
# DO NOT forget to disable this before training the model as it will generate A LOT of data
data_dump_dir = my_model.create_log_dir('dataset_dump')

# This is the directory where the dataset will be extracted
dataset_dir = create_user_dir('datasets/alexa')


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
    unknown_samples_batch:np.ndarray,
    seed:np.ndarray
) -> np.ndarray:
    """Augment a batch of audio clips and generate spectrograms

    This does the following, for each audio file path in the input batch:
    1. Read audio file
    2. Adjust its length to fit within the specified length
    3. Apply random augmentations to the audio sample using audiomentations
    4. Convert to the specified sample rate (if necessary)
    5. Generate a spectrogram from the augmented audio sample
    6. Dump the augmented audio and spectrogram (if necessary)

    NOTE: This will be execute in parallel across *separate* subprocesses.

    Arguments:
        path_batch: Batch of audio file paths
        label_batch: Batch of corresponding labels
        unknown_samples_batch: Batch of randomly selected "unknown" sample file paths
        seed: Batch of seeds to use for random number generation,
            This ensures that the "random" augmentations are reproducible

    Return:
        Generated batch of spectrograms from augmented audio samples
    """
    batch_length = path_batch.shape[0]
    height, width = frontend_settings.spectrogram_shape
    x_shape = (batch_length, height, 1, width)
    x_batch = np.empty(x_shape, dtype=np.int8)

    # This is the amount of padding we add to the beginning of the sample
    # This allows for "warming up" the noise reduction block
    padding_length_ms = 1000
    padded_frontend_settings = frontend_settings.copy()
    padded_frontend_settings.sample_length_ms += padding_length_ms

    # For each audio sample path in the current batch
    for i, (audio_path, labels, unknown_sample) in enumerate(zip(path_batch, label_batch, unknown_samples_batch)):
        class_id = np.argmax(labels)
        np.random.seed(seed[i])

        rn = np.random.random()
        use_cropped_sample_as_unknown = False
        using_silence_as_unknown = False

        # 30% of the time we want to replace this sample
        # either either silence or a cropped "known" sample
        if class_id == unknown_class_id and rn < 0.15:
            # 1% of the time we want to replace an "unknown" sample with silence
            if rn < .08:
                using_silence_as_unknown = True
                original_sample_rate = frontend_settings.sample_rate_hz
                sample = np.zeros((original_sample_rate,), dtype=np.float32)
            else:
                # Otherwise, find a "known" sample in the current batch
                # Later, we'll crop this sample and use it as an "unknown" sample
                choices = list(range(batch_length))
                np.random.shuffle(choices)
                for choice_index in choices:
                    if np.argmax(label_batch[choice_index]) >= unknown_class_id:
                        continue
                    audio_path = path_batch[choice_index]
                    use_cropped_sample_as_unknown = True
                    break

        if not using_silence_as_unknown:
            if class_id == unknown_class_id:
                audio_path = unknown_sample

            # Read the audio file
            try:
                sample, original_sample_rate = audio_utils.read_audio_file(audio_path, return_numpy=True, return_sample_rate=True)
            except Exception as e:
                raise RuntimeError(f'Failed to read: {audio_path}, err: {e}')


        # Create a buffer to hold the padded sample
        padding_length = int((original_sample_rate * padding_length_ms) / 1000)
        padded_sample_length = int((original_sample_rate * padded_frontend_settings.sample_length_ms) / 1000)
        padded_sample = np.zeros((padded_sample_length,), dtype=np.float32)

        # If we want to crop a "known" sample and use it as an unknown sample
        if use_cropped_sample_as_unknown:
            # Trim any silence from the sample
            trimmed_sample, _ = librosa.effects.trim(sample, top_db=15)
            # Randomly insert 20% to 40% of the trimmed sample into padded sample buffer
            # Note that the entire trimmed sample is actually added to the padded sample buffer
            # However, only the part of the sample that is after padding_length_ms will actually be used.
            # Everything before will eventually be dropped
            trimmed_sample_length = min(len(trimmed_sample), padded_sample_length)
            cropped_sample_percent = np.random.uniform(.2, .5)
            cropped_sample_length = int(trimmed_sample_length * cropped_sample_percent)
            if cropped_sample_length > .100 * original_sample_rate:
                # Add the beginning of the sample to the end of the padded sample buffer.
                # This simulates the sample streaming into the audio buffer,
                # but not being fully streamed in when an inference is invoked on the device.
                # In this case, we want the partial sample to be considered "unknown".
                padded_sample[-cropped_sample_length:] += trimmed_sample[:cropped_sample_length]
        else:
             # Otherwise, adjust the audio clip to the length defined in the frontend_settings
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
                audiomentations.Gain(min_gain_in_db=0.95, max_gain_in_db=1.5, p=1.0),
                audiomentations.AddBackgroundNoise(
                    f'{dataset_dir}/_background_noise_/brd2601',
                    min_absolute_rms_in_db=-75.0,
                    max_absolute_rms_in_db=-60.0,
                    noise_rms="absolute",
                    lru_cache_size=50,
                    p=1.0
                ),
                audiomentations.AddBackgroundNoise(
                    f'{dataset_dir}/_background_noise_/ambient',
                    min_snr_in_db=-2, # The lower the SNR, the louder the background noise
                    max_snr_in_db=35,
                    noise_rms="relative",
                    lru_cache_size=50,
                    p=0.95
                ),
                audiomentations.AddGaussianSNR(min_snr_in_db=30, max_snr_in_db=60, p=0.25),
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
            dtype=np.int8
        )

        # The input audio sample was padded with padding_length_ms of background noise
        # Drop the background noise from the final spectrogram used for training
        spectrogram = spectrogram[-height:, :]
        # The output spectrogram is 2D, add a channel dimension to make it 3D:
        # (height, width, channels=1)

        # Convert the spectrogram dimension from
        # <time, features> to
        # <time, 1, features>
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
            spectrogram_dumped = np.squeeze(spectrogram, axis=-2)
            # Transpose to put the time on the x-axis
            spectrogram_dumped = np.transpose(spectrogram_dumped)
            # Convert from int8 to uint8
            spectrogram_dumped = np.clip(spectrogram_dumped +128, 0, 255)
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
        self.all_unknown_samples = []
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


        # Download the synthetic "alexa" dataset and extract into the dataset directory
        download_verify_extract(
            url='https://www.dropbox.com/s/b6nd8xr7zzwmd6d/sl_synthetic_alexa.7z?dl=1',
            dest_dir=dataset_dir,
            file_hash='e657e91d6ea55639ce2e9a4dd8994c112fda2de0',
            show_progress=False,
            remove_root_dir=False,
            clean_dest_dir=False
        )

        # Download the synthetic alexa "unknown" dataset and extract into the dataset sub-directory: '_unknown'
        download_verify_extract(
            url='https://www.dropbox.com/s/86wh4defrqj0n9r/sl_synthetic_alexa_unknown.7z?dl=1',
            dest_dir=f'{dataset_dir}/_unknown',
            file_hash='2693e5fc72c52f199de2a69ed720644c2c363591',
            show_progress=False,
            remove_root_dir=False,
            clean_dest_dir=False
        )

        # Download the synthetic generic "unknown" dataset and extract into the dataset sub-directory: '_unknown'
        download_verify_extract(
            url='https://www.dropbox.com/s/zwvztg39a340b5q/sl_synthetic_generic_unknown.7z?dl=1',
            dest_dir=f'{dataset_dir}/_unknown',
            file_hash='6729b4763a506e427beb0909069219767f3d0d6f',
            show_progress=False,
            remove_root_dir=False,
            clean_dest_dir=False
        )

        # Download the mlcommons subset and extract into the dataset sub-directory: '_unknown/mlcommons_keywords'
        download_verify_extract(
            url='https://www.dropbox.com/s/j4p9w4h92e8rruo/mlcommons_keywords_subset_part1.7z?dl=1',
            dest_dir=f'{dataset_dir}/_unknown/mlcommons_keywords',
            file_hash='6f515d8247e2fee70cd0941420918c8fe57a31e8',
            show_progress=False,
            remove_root_dir=False,
            clean_dest_dir=False
        )

        # Download the mlcommons subset and extract into the dataset sub-directory: '_unknown/mlcommons_keywords'
        download_verify_extract(
            url='https://www.dropbox.com/s/zacujsccjgk92b2/mlcommons_keywords_subset_part2.7z?dl=1',
            dest_dir=f'{dataset_dir}/_unknown/mlcommons_keywords',
            file_hash='7816f5ffa1deeafa9b5b3faae563f44198031796',
            show_progress=False,
            remove_root_dir=False,
            clean_dest_dir=False
        )

        # Download the mlcommons voice and extract into the dataset sub-directory: '_unknown/mlcommons_voice'
        download_verify_extract(
            url='https://www.dropbox.com/s/l9uxyr22w3jgenc/common_voice_subset.7z?dl=1',
            dest_dir=f'{dataset_dir}/_unknown/mlcommons_voice',
            file_hash='ce424afd5d9b754f3ea6b3a4f78304f48e865f93',
            show_progress=False,
            remove_root_dir=False,
            clean_dest_dir=False
        )

        # Download the BRD2601 background microphone audio and add it to the _background_noise_/brd2601 of the dataset
        download_verify_extract(
            url='https://github.com/SiliconLabs/mltk_assets/raw/master/datasets/brd2601_background_audio.7z',
            dest_dir=f'{dataset_dir}/_background_noise_/brd2601',
            file_hash='3069A85002965A7830C660343C215EDD4FAE39C6',
            show_progress=False,
            remove_root_dir=False,
            clean_dest_dir=False,
        )


        # Download other ambient background audio and add it to the _background_noise_/ambient of the dataset
        # See https://mixkit.co/
        URLS = [
            'https://assets.mixkit.co/sfx/download/mixkit-very-crowded-pub-or-party-loop-360.wav',
            'https://assets.mixkit.co/sfx/download/mixkit-big-crowd-talking-loop-364.wav',
            'https://assets.mixkit.co/sfx/download/mixkit-restaurant-crowd-talking-ambience-444.wav',
            'https://assets.mixkit.co/sfx/download/mixkit-keyboard-typing-1386.wav',
            'https://assets.mixkit.co/sfx/download/mixkit-office-ambience-447.wav',
            'https://assets.mixkit.co/sfx/download/mixkit-hotel-lobby-with-dining-area-ambience-453.wav'
        ]

        for url in URLS:
            fn = os.path.basename(url)
            dst_path = f'{dataset_dir}/_background_noise_/ambient/{fn}'
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if not os.path.exists(dst_path):
                download_url(url=url, dst_path=dst_path)
                sample, original_sample_rate = audio_utils.read_audio_file(
                    dst_path,
                    return_sample_rate=True,
                    return_numpy=True
                )
                sample = audio_utils.resample(
                    sample,
                    orig_sr=original_sample_rate,
                    target_sr=frontend_settings.sample_rate_hz
                )
                audio_utils.write_audio_file(dst_path, sample, sample_rate=16000)


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

        # While training, the "unknown" class has a fixed size of samples
        # However, the actual number of "unknown" samples is much larger than the class size.
        # As such, we shuffle the unknown samples an randomly select from all of them while training.
        unknown_samples_ds = tf.data.Dataset.from_tensor_slices(self.all_unknown_samples)
        unknown_samples_ds = unknown_samples_ds.shuffle(max(len(self.all_unknown_samples), 10000), reshuffle_each_iteration=True)
        self.summary += f'{subset} subset shuffling {len(self.all_unknown_samples)} "unknown" samples\n'
        self.all_unknown_samples = []


        if subset:
            per_job_batch_multiplier = 1000
            per_job_batch_size = my_model.batch_size * per_job_batch_multiplier

            # We use an incrementing counter as the seed for the random augmentations
            # This helps to keep the training reproducible
            seed_counter = tf.data.experimental.Counter()
            features_ds = features_ds.zip((features_ds, labels_ds, unknown_samples_ds, seed_counter))

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
                dtype=np.int8,
                #n_jobs=84 if subset == 'training' else 32, # These are the settings for a 256 CPU core cloud machine
                n_jobs=72 if subset == 'training' else 32, # These are the settings for a 128 CPU core cloud machine
                #n_jobs=44 if subset == 'training' else 16, # These are the settings for a 96 CPU core cloud machine
                #n_jobs=50 if subset == 'training' else 25, # These are the settings for a 84 CPU core cloud machine
                #n_jobs=36 if subset == 'training' else 12, # These are the settings for a 64 CPU core cloud machine
                #n_jobs=28 if subset == 'training' else 16, # These are the settings for a 48 CPU core cloud machine
                #n_jobs=.65 if subset == 'training' else .35,
                #n_jobs=1,
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

        Then populate the "_unknown_" class with an empty list of length: unknown_class_multiplier * len(<alexa class>)
        The empty values will dynamically populated from randomly chosen values in the full "unknown" class list.

        """
        unknown_dir = f'{dataset_dir}/_unknown/unknown'
        mlcommons_keywords_dir = f'{dataset_dir}/_unknown/mlcommons_keywords'
        mlcommons_voice_dir = f'{dataset_dir}/_unknown/mlcommons_voice'

        # Create a list of all possible "unknown" samples
        file_list = list([f'_unknown/unknown/{x}' for x in os.listdir(unknown_dir) if x.endswith('.wav') and os.path.getsize(f'{unknown_dir}/{x}') > 0])

        # All all the mlcommons_keywords "unknown" samples that are not the "known" sample
        for kw in os.listdir(mlcommons_keywords_dir):
            if kw in my_model.classes:
                continue
            d = f'{mlcommons_keywords_dir}/{kw}'
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith('.wav'):
                    file_list.append(f'_unknown/mlcommons_keywords/{kw}/{fn}')

        # The ML commons voice dataset contain samples of people speaking sentences.
        # Determine how long each sample is and add it that many times to the file list
        # This way can randomly choice different parts of the sample
        for fn in os.listdir(mlcommons_voice_dir):
            if fn.endswith('.wav'):
                p = f'{mlcommons_voice_dir}/{fn}'
                multiplier = max(1, os.path.getsize(p) // (2 * 16000))
                for _ in range(multiplier):
                    file_list.append(f'_unknown/mlcommons_voice/{fn}')


        # Sort the unknown samples by "voice"
        # This helps to ensure voices are only present in a given subset
        file_list = sorted(file_list)
        file_list = shuffle_file_list_by_group(file_list, get_sample_group_id_from_path)

        # Split the file list for the current subset
        file_list = split_file_list(file_list, split)

        # Populate the "_unknown_" class with empty strings
        # The number of "_unknown_" entries is: <# of known samples> * unknown_class_multiplier
        # The empty strings are dynamically populated in audio_pipeline_with_augmentations()
        # with randomly selected "unknown" samples
        for key,value in sample_paths.items():
            if key != '_unknown_':
                sample_paths['_unknown_'] = [''] * int(len(value) * unknown_class_multiplier)
                break

        self.all_unknown_samples = [f'{directory}/{x}' for x in file_list]



def get_sample_group_id_from_path(p:str) -> str:
    """Extract the "voice hash" from the sample path.

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


    raise RuntimeError(f'Failed to get voice hash from {p}')


my_model.dataset = MyDataset()




#################################################
# Audio Classifier Settings
#
# These are additional parameters to include in
# the generated .tflite model file.
# The settings are used by the ble_audio_classifier app
# NOTE: Corresponding command-line options will override these values.

# This the amount of time in milliseconds between audio processing loops
# Since we're using the audio detection block, we want this to be as short as possible
my_model.model_parameters['latency_ms'] = 200
# The minimum number of inference results to average when calculating the detection value
my_model.model_parameters['minimum_count'] = 2
# Controls the smoothing.
# Drop all inference results that are older than <now> minus window_duration
# Longer durations (in milliseconds) will give a higher confidence that the results are correct, but may miss some commands
my_model.model_parameters['average_window_duration_ms'] = int(my_model.model_parameters['latency_ms']*my_model.model_parameters['minimum_count']*1.1)
# Define a specific detection threshold for each class
my_model.model_parameters['detection_threshold'] = int(.80*255)
# Amount of milliseconds to wait after a keyword is detected before detecting the SAME keyword again
# A different keyword may be detected immediately after
my_model.model_parameters['suppression_ms'] = 900
# Set the volume gain scaler (i.e. amplitude) to apply to the microphone data. If 0 or omitted, no scaler is applied
my_model.model_parameters['volume_gain'] = 0
# Enable verbose inference results
my_model.model_parameters['verbose_model_output_logs'] = False
# Uncomment this to increase the baud rate
# NOTE: You must use Simplicity Studio to increase the baud rate on the dev board as well
#my_model.model_parameters['baud_rate'] = 460800

##########################################################################################
# The following allows for running this model training script directly, e.g.:
# python keyword_spotting_alexa.py
#
# Note that this has the same functionality as:
# mltk train keyword_spotting_alexa
#
if __name__ == '__main__':
    from mltk import cli

    # Setup the CLI logger
    cli.get_logger(verbose=True)


    # If this is true then this will do a "dry run" of the model testing
    # If this is false, then the model will be fully trained
    test_mode_enabled = True

    # Train the model
    # This does the same as issuing the command:  mltk train keyword_spotting_alexa-test --clean)
    train_results = mltk_core.train_model(my_model, clean=True, test=test_mode_enabled)
    print(train_results)

    # Evaluate the model against the quantized .h5 (i.e. float32) model
    # This does the same as issuing the command: mltk evaluate keyword_spotting_alexa-test
    tflite_eval_results = mltk_core.evaluate_model(my_model, verbose=True, test=test_mode_enabled)
    print(tflite_eval_results)

    # Profile the model in the simulator
    # This does the same as issuing the command: mltk profile keyword_spotting_alexa-test
    profiling_results = mltk_core.profile_model(my_model, test=test_mode_enabled)
    print(profiling_results)
