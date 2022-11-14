# Model Operations

The following model operation APIs are available:

- [profile_model()](mltk.core.profile_model) - Profile a model for time and energy 
- [train_model()](mltk.core.train_model) - Train a model
- [evaluate_model()](mltk.core.evaluate_model) - Evaluate a trained model for accuracy 
- [update_model_parameters()](mltk.core.update_model_parameters) - Update a model's embedded parameters 
- [quantize_model()](mltk.core.quantize_model) - Quantize (i.e. compress) a trained model
- [view_model()](mltk.core.view_model) - View a model in an interactive diagram
- [summarize_model()](mltk.core.summarize_model) - Generate text summary of a model 


The source code for these APIs may be found on Github at [https://github.com/siliconlabs/mltk/tree/master/mltk/core](https://github.com/siliconlabs/mltk/tree/master/mltk/core).


## profile_model

```{eval-rst}
.. autofunction::  mltk.core.profile_model

.. autoclass:: mltk.core.profiling_results.ProfilingModelResults
   :members:

.. autoclass:: mltk.core.profiling_results.ProfilingLayerResult
   :members:
```

## train_model

```{eval-rst}
.. autofunction::  mltk.core.train_model
.. autoclass:: mltk.core.train_model.TrainingResults
   :members:
```

## evaluate_model

```{eval-rst}
.. autofunction::  mltk.core.evaluate_model
.. autoclass::  mltk.core.EvaluationResults
   :members:
```

### evaluate_classifier

```{eval-rst}
.. autofunction::  mltk.core.evaluate_classifier
.. autoclass::  mltk.core.ClassifierEvaluationResults
   :members:
```

### evaluate_autoencoder

```{eval-rst}
.. autofunction::  mltk.core.evaluate_autoencoder
.. autoclass::  mltk.core.AutoEncoderEvaluationResults
   :members:
```

## update_model_parameters

```{eval-rst}
.. autofunction::  mltk.core.update_model_parameters
```

## quantize_model

```{eval-rst}
.. autofunction::  mltk.core.quantize_model
```

## view_model

```{eval-rst}
.. autofunction::  mltk.core.view_model
```

## summarize_model

```{eval-rst}
.. autofunction::  mltk.core.summarize_model
```
