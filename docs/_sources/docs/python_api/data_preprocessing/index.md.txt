# Data Preprocessing

The following data pre-processing utilities are available:

- [Common Utilities](utilities.md) - Common utilities
- [AudioFeatureGenerator](audio_feature_generator.md) - Converts raw audio samples into spectrograms
- [ParallelImageDataGenerator](./image_data_generator.md) - Allows for efficiently generating augmented image samples during model training and evaluation
- [ParallelAudioDataGenerator](./audio_data_generator.md) - Allows for efficiently generating augments audio samples during model training and evaluation



The source code for these APIs may be found on Github at [https://github.com/siliconlabs/mltk/tree/master/mltk/core/preprocess](https://github.com/siliconlabs/mltk/tree/master/mltk/core/preprocess).




```{toctree}
:maxdepth: 1
:hidden:

./audio_feature_generator
./audio_feature_generator_settings
./image_data_generator
./image_data_generator_params
./audio_data_generator
./audio_data_generator_params
./utilities

```