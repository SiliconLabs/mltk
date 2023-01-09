__NOTE:__ Refer to the [online documentation](https://siliconlabs.github.io/mltk) to properly view this file
# Utilities

The MLTK Python package comes with various utility scripts.

The source code for these APIs may be found on Github at [https://github.com/siliconlabs/mltk/tree/master/mltk/utils](https://github.com/siliconlabs/mltk/tree/master/mltk/utils).



The following utilities are available:


<table class="autosummary longtable docutils align-default">
<colgroup>
<col style="width: 10%">
<col style="width: 90%">
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="audio_dataset_generator/index.html"><code class="xref py py-obj docutils literal notranslate"><span class="pre">mltk.utils.audio_dataset_generator</span></code></a></p></td>
<td><p>Allows for generating a synthetic keyword audio datasets</p></td>
</tr>
</tbody>
</table>

<table class="autosummary longtable docutils align-default">
<colgroup>
<col style="width: 10%">
<col style="width: 90%">
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="uart_stream/index.html"><code class="xref py py-obj docutils literal notranslate"><span class="pre">mltk.utils.uart_stream</span></code></a></p></td>
<td><p>Allows for streaming binary data between a Python script and embedded device via UART</p></td>
</tr>
</tbody>
</table>

<table class="autosummary longtable docutils align-default">
<colgroup>
<col style="width: 10%">
<col style="width: 90%">
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="jlink_stream/index.html"><code class="xref py py-obj docutils literal notranslate"><span class="pre">mltk.utils.jlink_stream</span></code></a></p></td>
<td><p>Allows for transferring binary data between a Python script and a JLink-enabled embedded device via the debug interface</p></td>
</tr>
</tbody>
</table>

<table class="autosummary longtable docutils align-default">
<colgroup>
<col style="width: 10%">
<col style="width: 90%">
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="serial_reader/index.html"><code class="xref py py-obj docutils literal notranslate"><span class="pre">mltk.utils.serial_reader</span></code></a></p></td>
<td><p>Allows for reading from a serial port (e.g. from the UART of an embedded device)</p></td>
</tr>
</tbody>
</table>



```{eval-rst}


.. autosummary::
   :toctree: archive_downloader
   :template: custom-module-template.rst

   mltk.utils.archive_downloader

.. autosummary::
   :toctree: archive
   :template: custom-module-template.rst

   mltk.utils.archive

.. autosummary::
   :toctree: bin2header
   :template: custom-module-template.rst

   mltk.utils.bin2header

.. autosummary::
   :toctree: cmake
   :template: custom-module-template.rst

   mltk.utils.cmake

.. autosummary::
   :toctree: gpu
   :template: custom-module-template.rst

   mltk.utils.gpu

.. autosummary::
   :toctree: hasher
   :template: custom-module-template.rst

   mltk.utils.hasher


.. autosummary::
   :toctree: logger
   :template: custom-module-template.rst

   mltk.utils.logger

.. autosummary::
   :toctree: path
   :template: custom-module-template.rst

   mltk.utils.path


.. autosummary::
   :toctree: python
   :template: custom-module-template.rst

   mltk.utils.python


.. autosummary::
   :toctree: shell_cmd
   :template: custom-module-template.rst

   mltk.utils.shell_cmd

.. autosummary::
   :toctree: string_formatting
   :template: custom-module-template.rst

   mltk.utils.string_formatting

.. autosummary::
   :toctree: system
   :template: custom-module-template.rst

   mltk.utils.system


.. autosummary::
   :toctree: signal_handler
   :template: custom-class-template.rst

   mltk.utils.signal_handler.SignalHandler

.. autosummary::
   :toctree: process_pool
   :template: custom-class-template.rst

   mltk.utils.process_pool.ProcessPool


```



```{toctree}
:maxdepth: 1
:hidden:

./archive_downloader
./archive
./bin2header
./cmake
./gpu
./hasher
./logger
./path
./python
./shell_cmd
./string_formatting
./system
./serial_reader/index
./signal_handler
./uart_stream/index
./jlink_stream/index
./process_pool
./process_pool_batch
./audio_dataset_generator/index
```