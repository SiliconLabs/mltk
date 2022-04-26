# Where is my trained model?

After [training](../guides/model_training.md) a model,
the trained model files (`.tflite` and `.h5`) are added to the `.mltk.zip` [model archive](../guides/model_archive.md) file
which is created in the same directory as the [model specification](../guides/model_specification.md) file.

Additionally, all intermediate training files can be found at: `~/.mltk/models/<model name>`   
where `<model name>` is the name of the trained file.

For example, say we have the [model specification](../guides/model_specification.md) file:

```
~/workspace/my_model.py
```

And we run the command:

```shell
cd ~/workspace
mltk train my_model
```

Then after training completes, we'll have:

```
~/workspace/my_model.py           <-- Model specification file
~/workspace/my_model.mltk.zip     <-- Model archive, contains trained .tflite and .h5 model files
```

And also, all intermediate training logs can be found at: `~/.mltk/models/my_model`

