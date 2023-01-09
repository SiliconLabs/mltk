__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# UART Stream

This allows for streaming binary data between a Python script and embedded device via UART.

Features:
- Asynchronous reception of binary data
- Data flow control
- C++ library (see [__mltk repo__/cpp/shared/uart_stream](https://github.com/siliconlabs/mltk/blob/master/cpp/shared/uart_stream))
- Send/receive "commands"

__NOTE:__ The embedded device must be running the [uart_stream](https://github.com/siliconlabs/mltk/blob/master/cpp/shared/uart_stream) C++ library for this Python package to work.


## Example Usage

The following is a snippet taken from [alexa_demo.py](https://github.com/siliconlabs/mltk/blob/master/cpp/shared/apps/audio_classifier/python/alexa_demo/alexa_demo.py)

```python
import io
from mltk.utils.uart_stream import UartStream

with UartStream() as uart:
   data_buffer = io.BytesIO()

   while True:
      cmd = uart.read_command()

      if cmd.code == 1:
         print(f'Command received: {data_buffer.tell()} bytes')
         uart.flush_input()
         data_buffer.seek(0)
         return data_buffer

      data = uart.read()
      if not data:
         uart.wait(0.100)
         continue

      if data_buffer.getbuffer().nbytes == 0:
            print('Receiving command ...')

      data_buffer.write(data)

```


Also see [UART Stream Data Test](https://siliconlabs.github.io/mltk/cpp/shared/uart_stream/examples/data_test/README.html)


## API Reference

```{eval-rst}

.. autosummary::
   :toctree: uart_stream
   :template: custom-class-template.rst

   mltk.utils.uart_stream.UartStream

```



```{toctree}
:maxdepth: 1
:hidden:

./uart_stream
```