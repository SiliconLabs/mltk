__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# Model Operations


The following model operation APIs are available:

- [profile_model()](profile.md) - Profile a model for time and energy 
- [train_model()](train.md) - Train a model
- [evaluate_model()](evaluate.md) - Evaluate a trained model for accuracy 
- [update_model_parameters()](update_model_parameters.md) - Update a model's embedded parameters 
- [quantize_model()](quantize.md) - Quantize (i.e. compress) a trained model
- [view_model()](view.md) - View a model in an interactive diagram
- [summarize_model()](summarize.md) - Generate text summary of a model 


The source code for these APIs may be found on Github at [https://github.com/siliconlabs/mltk/tree/master/mltk/core](https://github.com/siliconlabs/mltk/tree/master/mltk/core).



```{toctree}
:maxdepth: 1
:hidden:

./train
./profile
./evaluate
./quantize
./summarize
./view
./update_model_parameters
```