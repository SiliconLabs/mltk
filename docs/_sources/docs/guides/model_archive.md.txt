# Model Archive File

The MLTK uses an archive file (`.mltk.zip`) to store the relevant model information.

## Overview

The model archive file is automatically created after running the [train](./model_training.md) command 
and is updated after running the [evaluate](./model_evaluation.md), [quantize](./model_quantization.md), and [update_params](./model_parameters.md) commands.

The model archive file uses the standard [Zip File Format](https://docs.fileformat.com/compression/zip)  
and its name has the format: `<model name>.mltk.zip` where `<model name>` is the name of the MLTK model.

The model archive file is useful as it allows for grouping the various training and evaluation files into a single, distributable file.  
This file can also be directly loaded by many MLTK commands and Python APIs, e.g.:

```shell
mltk profile ~/my_model.mltk.zip
```


## Contents

The model archive file stores a given model's:  
- Model specification Python script
- Trained model files (`.tflite`, `.h5`)
- Training logs
- Evaluation logs


## Directory Structure

Assume we have the following model archive file `~/workspace/my_model.mltk.zip`.
The contents of this archive would have the following contents:

```shell
/my_model.py                   - The model specification script
/my_model.tflite               - The quantized model which can programmed onto an embedded device
/my_model.h5                   - The trained, non-quantized, Keras model
/my_model.h5.summary.txt       - A text summary of the .h5 model
/my_model.tflite.summary.txt   - A text summary of the .tflite model
/train/log.txt                 - Log file generated during training
/train/training-history.png    - Training history diagram
/train/training-history.json   - Training history in JSON format
/eval/h5/                      - Evaluation results from the .h5 (i.e. non-quantized) model
/eval/tflite/                  - Evaluation results from the .tflite (i.e. quantized) model
```






