__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# J-Link Stream

This allows for transferring binary data between a Python script and a JLink-enabled embedded device via the debug interface.


__NOTE:__ The embedded device must be running the [jlink_stream](https://github.com/siliconlabs/mltk/blob/master/cpp/shared/jlink_stream) C++ library for this Python package to work.


## Example Usage


The following is a snippet taken from [classify_audio_mltk_cli.py](https://github.com/sldriedler/mltk/blob/master/mltk/cli/classify_audio_mltk_cli.py#L474)

```python
from mltk.utils.jlink_stream import (JlinkStream, JLinkDataStream)

with JlinkStream() as jlink_stream:
   audio_stream = jlink_stream.open('audio', mode='r')
   chunk_data = audio_stream.read_all(audio_stream.read_data_available)

```


## API Reference

```{eval-rst}

.. autosummary::
   :toctree: jlink_stream
   :template: custom-class-template.rst

   mltk.utils.jlink_stream.JlinkStream

.. autosummary::
   :toctree: data_stream
   :template: custom-class-template.rst

   mltk.utils.jlink_stream.JLinkDataStream

.. autosummary::
   :toctree: command_stream
   :template: custom-class-template.rst

   mltk.utils.jlink_stream.JlinkCommandStream

.. autosummary::
   :toctree: stream_options
   :template: custom-class-template.rst

   mltk.utils.jlink_stream.JlinkStreamOptions


```



```{toctree}
:maxdepth: 1
:hidden:

./jlink_stream
./data_stream
./command_stream
./stream_options
```