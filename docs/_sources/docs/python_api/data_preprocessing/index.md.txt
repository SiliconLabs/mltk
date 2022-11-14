# Data Preprocessing


The following data pre-processing utilities are available:

- [ParallelImageDataGenerator](./image_data_generator.md) - Allows for efficiently generating augmented image samples during model training and evaluation
- [ParallelAudioDataGenerator](./audio_data_generator.md) - Allows for efficiently generating augments audio samples during model training and evaluation
- [AudioFeatureGenerator](audio_feature_generator.md) - Converts raw audio samples into spectrograms

The source code for these APIs may be found on Github at [https://github.com/siliconlabs/mltk/tree/master/mltk/core/preprocess](https://github.com/siliconlabs/mltk/tree/master/mltk/core/preprocess).




```{toctree}
:maxdepth: 1
:hidden:

./image_data_generator
./audio_data_generator
./audio_feature_generator

```