""" # Training Data Pre-processing

This contains helpers to generate data during model training.
Given a dataset of raw samples, this will potentially augment then provide
batches of 'training' and 'evaluation' samples to the Tensorflow training scripts.

The following data generators are available:

- `audio.ParallelAudioDataGenerator` - Generate audio training data 
- `image.ParallelImageDataGenerator` - Generate image training data

"""
