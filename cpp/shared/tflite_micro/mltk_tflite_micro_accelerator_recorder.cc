#ifndef MLTK_DLL_IMPORT

#include "mltk_tflite_micro_accelerator_recorder.hpp"


namespace mltk 
{


TfliteMicroAcceleratorRecorder& TfliteMicroAcceleratorRecorder::instance()
{
    static uint8_t instance_buffer[sizeof(TfliteMicroAcceleratorRecorder)];
    static TfliteMicroAcceleratorRecorder* instance_ptr = nullptr;

    if(instance_ptr == nullptr)
    {
        instance_ptr = new(instance_buffer)TfliteMicroAcceleratorRecorder();
    }

    return *instance_ptr;
}

bool TfliteMicroAcceleratorRecorder::is_enabled()
{
    auto& self = instance();
    return self._enabled;
}

void TfliteMicroAcceleratorRecorder::set_enabled(bool enabled)
{
    auto& self = instance();
    if(enabled)
    {
        TfliteMicroRecorder::set_enabled(true);
        TfliteMicroRecorder::set_layer_callback(
            TfliteMicroRecorder::Event::ExecutionBegin, 
            TfliteMicroAcceleratorRecorder::_on_layer_execution_started
        );
        TfliteMicroRecorder::set_layer_callback(
            TfliteMicroRecorder::Event::ExecutionEnd, 
            TfliteMicroAcceleratorRecorder::_on_layer_execution_ending
        );
    }
    else 
    {
        TfliteMicroRecorder::set_layer_callback(
            TfliteMicroRecorder::Event::ExecutionBegin, 
            nullptr
        );
        TfliteMicroRecorder::set_layer_callback(
            TfliteMicroRecorder::Event::ExecutionEnd, 
            nullptr
        );
    }
    self._enabled = enabled;
}

bool TfliteMicroAcceleratorRecorder::is_program_recording_enabled()
{
    auto& self = instance();
    return self._program_recording_enabled;
}

void TfliteMicroAcceleratorRecorder::set_program_recording_enabled(bool enabled)
{
    auto& self = instance();
    if(enabled)
    {
        set_enabled(true);
    }
    self._program_recording_enabled = enabled;
}

msgpack_context_t* TfliteMicroAcceleratorRecorder::programs_context()
{
    auto& self = instance();
    return self._programs_context;
}

void TfliteMicroAcceleratorRecorder::reset()
{
    auto& self = instance();
    msgpack_buffered_writer_deinit(self._programs_context, true);
    self._programs_context = nullptr;
    self._program_active = false;
}


bool TfliteMicroAcceleratorRecorder::record_data(
    const char* key, 
    const void* data, 
    uint32_t length
)
{
    auto& self = instance();

    if(!self._enabled)
    {
        return false;
    }

    auto msgpack = TfliteMicroRecorder::get_context();
    msgpack_write_dict_bin(msgpack, key, data, length);

    return true;
}

bool TfliteMicroAcceleratorRecorder::record_data(
    const char* key, 
    const uint8_t* data, 
    uint32_t length
)
{
    auto& self = instance();

    if(!self._enabled)
    {
        return false;
    }

    auto msgpack = TfliteMicroRecorder::get_context();
    msgpack_write_dict_bin(msgpack, key, data, length);

    return true;
}

bool TfliteMicroAcceleratorRecorder::record_program(const void* data, unsigned length)
{
    auto& self = instance();

    if(self._programs_context == nullptr)
    {
        return false;
    }

    _start_program();
    msgpack_write_dict_bin(self._programs_context, "data", data, length);
    msgpack_finalize_dynamic(self._programs_context); // finalize the program dict
    self._program_active = false;

    return true;
}

void TfliteMicroAcceleratorRecorder::_on_layer_execution_started()
{
    auto& self = instance();

    self._program_active = false;

    if(self._program_recording_enabled)
    {
        assert(self._programs_context == nullptr);
        msgpack_buffered_writer_init(&self._programs_context, 4096);
        msgpack_write_array_marker(self._programs_context, -1);
    }
}

void TfliteMicroAcceleratorRecorder::_on_layer_execution_ending()
{
    auto& self = instance();
    
    if(self._programs_context != nullptr)
    {
        if(msgpack_finalize_dynamic(self._programs_context) == 0) // finalize the programs array
        {
            auto msgpack = TfliteMicroRecorder::get_context();
            msgpack_write_dict_context(msgpack, "programs", self._programs_context);
        }

        msgpack_buffered_writer_deinit(self._programs_context, true);
        self._programs_context = nullptr;
        self._program_active = false;
    }
}

void TfliteMicroAcceleratorRecorder::_start_program()
{
    auto& self = instance();
    
    if(self._programs_context != nullptr && !self._program_active)
    {
        self._program_active = true;
        msgpack_write_dict_marker(self._programs_context, -1);
    }
}

} // namespace mltk 

#endif // #ifndef MLTK_DLL_IMPORT