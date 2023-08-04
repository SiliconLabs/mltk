__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# Tutorials


The following tutorials provide end-to-end guides on how to develop machine learning model using the MLTK:

| Name                                                                                                                                      | Description                                                                                                                                                                                            |
|-------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Keyword Spotting - On/Off](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_on_off.html)                               | Develop an ML model to detect the keywords: "on" or "off"                                                                                                                                              |
| [Keyword Spotting - Pac-Man](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_pacman.html)                              | Develop a demo to play the game Pac-Man in a web browser using the keywords: "Left", "Right", "Up", "Down", "Stop", "Go"                                                                               |
| [Keyword Spotting - Alexa](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_alexa.html)                                 | Develop a demo to issue "Alexa" commands to the [AVS](https://developer.amazon.com/en-US/docs/alexa/alexa-voice-service/get-started-with-alexa-voice-service.html) cloud and locally play the response |
| [Image Classification - Rock/Paper/Scissors](https://siliconlabs.github.io/mltk/mltk/tutorials/image_classification.html)                 | Develop an image classification ML model to detect the hand gestures: "rock", "paper", "scissors"                                                                                                      |
| [Model Training in the "Cloud"](https://siliconlabs.github.io/mltk/mltk/tutorials/cloud_training_with_vast_ai.html)                       | _Vastly_ improve model training times by training a model in the "cloud" using [vast.ai](http://vast.ai)                                                                                               |
| [Logging to the Cloud](https://siliconlabs.github.io/mltk/mltk/tutorials/cloud_logging_with_wandb.html)                                   | Log model files and metrics to the cloud during training and evaluation using [Weights & Biases](http://wandb.ai)                                                                                      |
| [Model Optimization for MVP Hardware Accelerator](https://siliconlabs.github.io/mltk/mltk/tutorials/model_optimization.html)              | Use the various MLTK tools to optimize a model to fit within an embedded device's resource constraints                                                                                                 |
| [Keyword Spotting with Transfer Learning](https://siliconlabs.github.io/mltk/mltk/tutorials/keyword_spotting_with_transfer_learning.html) | Use a pre-trained model to quickly train a new model that detects the keywords: "one", "two", "three", "four"                                                                                          |
| [Fingerprint Authentication](https://siliconlabs.github.io/mltk/mltk/tutorials/fingerprint_authentication.html)                           | Use ML to generate unique signatures from images of fingerprints to authenticate users                                                                                                                 |
| [ONNX to TF-Lite Model Conversion](https://siliconlabs.github.io/mltk/mltk/tutorials/onnx_to_tflite.html)                                 | Describes how to convert an [ONNX](https://onnx.ai/) formatted model file to the `.tflite` model format required by embedded targets                                                                   |
| [Model Debugging](https://siliconlabs.github.io/mltk/mltk/tutorials/model_debugging.html)                                                 | Describes how to debug an ML model during training                                                                                                                                                     |
| [Add an Existing Script to the MLTK](https://siliconlabs.github.io/mltk/mltk/tutorials/add_existing_script_to_mltk.html)                  | Describes how to convert an existing Tensorflow training script to support the MLTK training flow                                                                                                      |
| [Synthetic Audio Dataset Generation](https://siliconlabs.github.io/mltk/mltk/tutorials/synthetic_audio_dataset_generation.html)           | Describes how to generate a custom audio dataset using synthetic data. This allows for training keyword spotting ML models with custom keywords                                                        |
| [Model Quantization Tips](https://siliconlabs.github.io/mltk/mltk/tutorials/model_quantization_tips.html)                                 | Provides tips on how to gain better quantization for your model                                                                                                                                        |
| [Quantized LSTM](https://siliconlabs.github.io/mltk/mltk/tutorials/quantized_lstm.html)                                                   | Describes how to create a quantized keyword spotting model with an LSTM layer                                                                                                                          |


```{eval-rst}
.. toctree::
   :maxdepth: 1
   :hidden:

   /mltk/tutorials/keyword_spotting_on_off
   /mltk/tutorials/keyword_spotting_pacman
   /mltk/tutorials/keyword_spotting_alexa
   /mltk/tutorials/image_classification
   /mltk/tutorials/cloud_training_with_vast_ai
   /mltk/tutorials/cloud_logging_with_wandb
   /mltk/tutorials/model_optimization
   /mltk/tutorials/keyword_spotting_with_transfer_learning
   /mltk/tutorials/fingerprint_authentication
   /mltk/tutorials/onnx_to_tflite
   /mltk/tutorials/model_debugging
   /mltk/tutorials/add_existing_script_to_mltk
   /mltk/tutorials/synthetic_audio_dataset_generation
   /mltk/tutorials/model_quantization_tips
   /mltk/tutorials/quantized_lstm
```