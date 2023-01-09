__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# Serial Reader

This allows for reading from a serial port (e.g. from the UART of an embedded device).


## Example Usage

The following is a snippet taken from [profile_model.py:profile_model_on_device](https://github.com/siliconlabs/mltk/blob/master/mltk/core/profile_model.py#L146)


```python
import re
from mltk.utils.serial_reader import SerialReader

with SerialReader(
   start_regex=[
      re.compile('.*Starting Model Profiler', re.IGNORECASE),
      re.compile('Loading model', re.IGNORECASE)
   ],
   stop_regex=[re.compile(r'.*done.*', re.IGNORECASE)],
   fail_regex=[
      re.compile(r'.*hardfault.*', re.IGNORECASE),
      re.compile(r'.*error.*', re.IGNORECASE),
      re.compile(r'.*failed to alloc memory.*', re.IGNORECASE)
   ]
) as serial_reader:
   # Wait for up to a minute for the profiler to complete
   if not serial_reader.read(timeout=60):
      raise TimeoutError('Timed-out waiting for profiler on device to complete')

   # Check if the profiler failed
   if serial_reader.failed:
      raise RuntimeError(f'Profiler failed on device, err: {serial_reader.error_message}')

   # Retrieve the captured data
   device_log = serial_reader.captured_data
   print(device_log)

```


## API Reference

```{eval-rst}

.. autosummary::
   :toctree: serial_reader
   :template: custom-class-template.rst

   mltk.utils.serial_reader.SerialReader
```



```{toctree}
:maxdepth: 1
:hidden:

./serial_reader
```