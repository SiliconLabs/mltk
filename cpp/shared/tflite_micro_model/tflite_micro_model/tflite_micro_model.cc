#include <new>
#include <cstdlib>
#include "em_device.h"

#include "tensorflow/lite/schema/schema_generated.h"
#include "cpputils/string.hpp"
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "tflite_micro_model/tflite_micro_utils.hpp"
#include "mltk_tflite_micro_helper.hpp"
#include "mltk_tflite_micro_context.hpp"


#ifdef __arm__
#if !defined(__HEAP_SIZE)
#undef __HEAP_SIZE
extern "C" uint32_t __heap_size;
#define __HEAP_SIZE ((uint32_t)(&__heap_size))
#endif // ifndef __HEAP_SIZE
#endif // __arm__

namespace mltk
{

int adjust_required_tensor_arena_bytes_from_64bit_to_32bit(
    const void* flatbuffer,
    tflite::MicroInterpreter* interpreter,
    int tensor_arena_size
);

TfliteMicroModel::~TfliteMicroModel()
{
    unload();
}


bool TfliteMicroModel::load(
    const void* flatbuffer,
    const tflite::MicroOpResolver& op_resolver,
    uint8_t *runtime_buffer,
    int32_t runtime_buffer_size
)
{
    uint8_t* local_buffers[1] = {runtime_buffer};
    const int32_t local_buffer_sizes[1] = {runtime_buffer_size};

    return load(flatbuffer, &op_resolver, local_buffers, local_buffer_sizes, 1);
}

bool TfliteMicroModel::load(
    const void* flatbuffer, 
    const tflite::MicroOpResolver* op_resolver,
    uint8_t* buffers[],
    const int32_t buffer_sizes[],
    int32_t buffer_count
)
{
    if(buffer_count <= 0 || buffers == nullptr || buffer_sizes == nullptr)
    {
        MLTK_ERROR("Invalid buffer argument");
        return false;
    }

    int32_t allocated_buffer_size = 0;
    uint8_t* local_buffers[buffer_count];
    int32_t local_buffer_sizes[buffer_count];
    // The first buffer in the list is considered the "runtime buffer" 
    // which is also used to store persistent data and temp allocations
    uint8_t* runtime_buffer = buffers[0]; 
    int32_t runtime_buffer_size = buffer_sizes[0];

    for(int i = 0; i < buffer_count; ++i)
    {
        local_buffers[i] = buffers[i];
        local_buffer_sizes[i] = buffer_sizes[i];
    }

    // Ensure kernel messages are enabled while the model is loaded
    mltk::TfliteMicroKernelMessages::set_enabled(true);

    // Load the model's parameters from the metadata (if available)
    load_model_parameters(flatbuffer);

    // If no runtime buffer was specified,
    // then we need to allocate one now
    if(runtime_buffer == nullptr)
    {
        // If the runtime_buffer_size >=0, then we attempt to allocate a buffer now.
        // If the runtime_buffer_size  < 0, then we skip down to finding the optimal run-time buffer size.
        int model_runtime_size = 0;
        if(runtime_buffer_size >= 0)
        {
            // If the runtime_buffer_size == 0,
            // then attempt to retrieve the size from the .tflite model parameters
            if(runtime_buffer_size == 0)
            {
                parameters.get("runtime_memory_size", model_runtime_size);

                // If the size in the .tflite model parameters is < 0,
                // then we just skip to finding the optimal buffer size
                if(model_runtime_size < 0)
                {
                    model_runtime_size = 0;
                }
                else
                {
                    MLTK_INFO("Runtime memory size from .tflite model: %d", model_runtime_size);
                }
            }
            else
            {
                // Otherwise, allocate the size given as an argument to this API
                model_runtime_size = runtime_buffer_size;
            }


            // If we should attempt to allocate a buffer before searching for the optimal size
            if(model_runtime_size > 0)
            {
                allocated_buffer_size = model_runtime_size;


                #if INTPTR_MAX == INT64_MAX
                // The buffer size embedded into the .tflite is meant for a 32-bit ARM MCU
                // If we're running on a 64-bit system then add 1MB of additional memory
                // to account for 64-bit pointer overhead
                allocated_buffer_size += 1024*1024;
                #endif

                // Allocate the buffer with the specified size
                runtime_buffer = static_cast<uint8_t*>(malloc(allocated_buffer_size));
                local_buffers[0] = runtime_buffer;
                local_buffer_sizes[0] = allocated_buffer_size;

                if(runtime_buffer == nullptr)
                {
                    // If there isn't enough memory to allocate the buffer
                    // then fallback to the buffer size optimization algorithm
                    MLTK_WARN("Failed to allocate buffer with size: %d specified in .tflite model", allocated_buffer_size);
                }
                // Load the model with the buffer
                else if(load_interpreter(
                    flatbuffer, 
                    op_resolver, 
                    local_buffers, 
                    local_buffer_sizes,
                    buffer_count
                ))
                {
                    // If we successfully loaded the model with the runtime memory size from the flatbuffer
                    // then we're done
                    runtime_buffer_size = model_runtime_size;
                }
                else
                {
                    // We failed to load the interpreter with the given buffer
                    // so clean the buffer up
                    free(runtime_buffer);
                    runtime_buffer = nullptr;

                    // If the model has unsupported layers,
                    // then just return as we cannot load it
                    if(TfliteMicroKernelMessages::unknown_layers_detected())
                    {
                        MLTK_WARN("Model contains one or more unknown layers. You may need to update your OpResolver");
                        unload();
                        return false;
                    }

                    // Otherwise, if the specified buffer size was too small
                    // Then fallback to the buffer size optimization algorithm

                    MLTK_WARN("Failed to load model with buffer of size %d", model_runtime_size);
                }

                // If a size was specified by the API call and we failed to load the model with it
                // then just return the error
                // NOTE: If a size was specified in the model parameters and it was too small,
                //       then we just fallback to finding the optimal size below
                if(runtime_buffer_size > 0 && runtime_buffer == nullptr)
                {
                    unload();
                    return false;
                }
            }
        }

        // If the .tflite doesn't specify the runtime memory buffer size
        // or we failed to load the model because the buffer was too small
        // then use the buffer size optimization algorithm now
        if(runtime_buffer == nullptr)
        {
            local_buffers[0] = nullptr;
            local_buffer_sizes[0] = runtime_buffer_size;

            // Find the optimal buffer size
            if(!find_optimal_buffer_size(
                flatbuffer, 
                op_resolver, 
                local_buffers,
                local_buffer_sizes,
                buffer_count,
                runtime_buffer_size
            ))
            {
                // On failure, just return
                if(TfliteMicroKernelMessages::unknown_layers_detected())
                {
                    MLTK_WARN("Model contains one or more unknown layers. You may need to update your OpResolver");
                }
                else
                {
                    MLTK_ERROR("Failed to allocate buffer for model (due to heap memory overflow or the model is not fully supported)");
                }
                unload();
                return false;
            }

            allocated_buffer_size = runtime_buffer_size;

            #if INTPTR_MAX == INT64_MAX
            // If we're running on a 64-bit system then add 1MB of additional memory
            // to account for 64-bit pointer overhead.
            // This is only used when allocating temporary tenesors (e.g. context.GetTensor())
            allocated_buffer_size += 1024*1024;
            #endif

            // Allocate the buffer
            runtime_buffer = static_cast<uint8_t*>(malloc(allocated_buffer_size));
            local_buffers[0] = runtime_buffer;
            local_buffer_sizes[0] = allocated_buffer_size;

            if(runtime_buffer == nullptr)
            {
                // If this fails, something is wrong with find_optimal_buffer_size()
                MLTK_WARN("Failed to allocate buffer with size: %d", allocated_buffer_size);
                unload();
                return false;
            }
            // Load the model with the buffer
            else if(!load_interpreter(
                flatbuffer, 
                op_resolver, 
                local_buffers, 
                local_buffer_sizes,
                buffer_count
            ))
            {
                // If this fails, something is wrong with find_optimal_buffer_size()
                MLTK_WARN("Failed to allocate buffer with size: %d", allocated_buffer_size);
                unload();
                return false;
            }
        }

        // At this point, the runtime memory buffer is allocated and the model flatbuffer is loaded.
        // This model now owns the buffer
        // When the model is unloaded, the buffer will be automatically freed
        _runtime_buffer = runtime_buffer;
    }
    // Else if a pre-allocated buffer was given to this API
    else
    {
        allocated_buffer_size = runtime_buffer_size;
        local_buffers[0] = runtime_buffer;
        local_buffer_sizes[0] = allocated_buffer_size;

        // Verify the runtime_buffer_size is > 0
        if(runtime_buffer_size <= 0)
        {
            MLTK_ERROR("Must specify runtime_buffer_size when providing pre-allocated buffer");
            unload();
            return false;
        }
        // If a runtime buffer was provided, then load the interpreter with it now
        else if(!load_interpreter(
            flatbuffer, 
            op_resolver, 
            local_buffers, 
            local_buffer_sizes,
            buffer_count
        ))
        {
            // Return the error on failure
            if(TfliteMicroKernelMessages::unknown_layers_detected())
            {
                MLTK_WARN("Model contains one or more unknown layers. You may need to update your OpResolver");
            }
            else
            {
                MLTK_ERROR("Failed to load model with runtime buffer size: %d", runtime_buffer_size);
            }
            unload();
            return false;
        }
    }

    #ifdef __arm__
    _model_details._runtime_memory_size = runtime_buffer_size;
    #else
    _model_details._runtime_memory_size = adjust_required_tensor_arena_bytes_from_64bit_to_32bit(flatbuffer, _interpreter, runtime_buffer_size);
    #endif

    MLTK_RECORD_PARAM("total_runtime_size", (uint32_t)_interpreter->allocator_.used_bytes());
    MLTK_RECORD_PARAM("total_persistent_runtime_size", 
        (_interpreter->allocator_.persistent_buffer_allocator_ != nullptr) ?
            (uint32_t)_interpreter->allocator_.persistent_buffer_allocator_->GetPersistentUsedBytes() : 0
    );

    auto accelerator = mltk_tflite_micro_get_registered_accelerator();
    if(accelerator != nullptr)
    {
        _model_details._accelerator = accelerator->name();
        #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
        accelerator->set_simulator_memory("sram", runtime_buffer, allocated_buffer_size); // Be sure to use the actually allocated buffer size used by this 64-bit program
        accelerator->set_simulator_memory("flash", (void*)flatbuffer, 2*1024*1024);
        #endif
    }

    TfliteMicroModelHelper::set_processing_callback(
        tflite_context(), 
        this->_processing_callback, 
        this->_processing_callback_arg
    );

    return true;
}

void TfliteMicroModel::unload()
{
    auto accelerator = mltk_tflite_micro_get_registered_accelerator();

    if(accelerator != nullptr)
    {
        accelerator->deinit(&_interpreter->context_);
    }

    _flatbuffer = nullptr;
    _ops_resolver = nullptr;
    parameters.unload();
    _model_details.unload();
    MLTK_RECORD_RESET();
    if(_interpreter != nullptr)
    {
        _interpreter->~MicroInterpreter();
        _interpreter = nullptr;
    }

    if(_runtime_buffer != nullptr)
    {
        free(_runtime_buffer);
        _runtime_buffer = nullptr;
    }

    TfliteMicroModelHelper::set_active_tflite_context(nullptr);
}

bool TfliteMicroModel::is_loaded() const
{
    return _interpreter != nullptr;
}

bool TfliteMicroModel::invoke() const
{
    bool retval;
    auto accelerator = mltk_tflite_micro_get_registered_accelerator();

    if(!is_loaded())
    {
        MLTK_ERROR("Model not loaded");
        return false;
    }

    TfliteMicroModelHelper::set_active_tflite_context(tflite_context());

    if(profiler_is_enabled())
    {
        profiling::reset(this->profiler());
    }

    #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
    if(accelerator != nullptr)
    {
        retval = accelerator->invoke_simulator([this]() -> bool
        {
            return _interpreter->Invoke() == kTfLiteOk;
        });
    }
    else
    {
        retval = (_interpreter->Invoke() == kTfLiteOk);
    }
    #else
    retval = (_interpreter->Invoke() == kTfLiteOk);
    #endif

    TfliteMicroModelHelper::set_active_tflite_context(nullptr);

    return retval;
}

const TfliteMicroModelDetails& TfliteMicroModel::details() const
{
    return _model_details;
}

void TfliteMicroModel::print_summary(logging::Logger *logger) const
{
    auto& l = (logger != nullptr) ? *logger : get_logger();

    if(!is_loaded())
    {
        l.error("Model not loaded");
        return;
    }

    auto& input = *this->input(0);
    auto& output = *this->output(0);

    char fmt_buffer[32];
    const auto orig_flags = l.flags();
    l.flags(logging::Newline);

    l.info("Model details:");
    l.info("Name: %s", _model_details.name());
    l.info("Version: %d", _model_details.version());
    l.info("Date: %s", _model_details.date());
    l.info("Hash: %s", _model_details.hash());
    l.info("Accelerator: %s", _model_details.accelerator());

    l.info("Tensor runtime memory: %s", cpputils::format_units(_model_details.runtime_memory_size(), 3, fmt_buffer));;
    l.info("Input: %s", input.to_str());
    l.info("Output: %s", output.to_str());

    const auto& classes = _model_details.classes();
    const auto class_count = classes.size();
    if(class_count > 0)
    {
        l.flags().clear(logging::Newline);
        l.info("Classes: ");
        for(int i = 0; i < class_count; ++i)
        {
            l.info((i < class_count-1) ? "%s, " : "%s\n", classes[i]);
        }
        l.flags().set(logging::Newline);
    }

    if(*_model_details.description() != 0)
    {
        l.info("Description: %s", _model_details.description());
    }

    l.flags(orig_flags);
}

unsigned TfliteMicroModel::input_size() const
{
    if(!is_loaded())
    {
        MLTK_ERROR("Model not loaded");
        return false;
    }

    return _interpreter->inputs_size();
}

TfliteTensorView* TfliteMicroModel::input(unsigned index) const
{
    if(!is_loaded())
    {
        MLTK_ERROR("Model not loaded");
        return nullptr;
    }

    return reinterpret_cast<TfliteTensorView*>(_interpreter->input(index));
}

unsigned TfliteMicroModel::output_size() const
{
    if(!is_loaded())
    {
        MLTK_ERROR("Model not loaded");
        return false;
    }

    return _interpreter->outputs_size();
}

TfliteTensorView* TfliteMicroModel::output(unsigned index) const
{
    if(!is_loaded())
    {
        MLTK_ERROR("Model not loaded");
        return nullptr;
    }

    return reinterpret_cast<TfliteTensorView*>(_interpreter->output(index));
}

bool TfliteMicroModel::enable_profiler()
{
    #ifdef TFLITE_MICRO_PROFILER_ENABLED
    if(is_loaded())
    {
        MLTK_ERROR("Model already loaded");
        return false;
    }
    TfliteMicroProfiler::set_enabled(true);
    return true;
    #else
    MLTK_ERROR("C++ library not build with profiling support");
    return false;
    #endif
}

bool TfliteMicroModel::profiler_is_enabled() const
{
    return TfliteMicroProfiler::is_enabled();
}

profiling::Profiler* TfliteMicroModel::profiler() const
{
    return profiling::get("Inference");
}

bool TfliteMicroModel::enable_recorder()
{
    #if TFLITE_MICRO_RECORDER_ENABLED
    if(is_loaded())
    {
        MLTK_ERROR("Model already loaded");
        return false;
    }
    TfliteMicroRecorder::set_enabled(true);
    return true;
    #else
    MLTK_ERROR("C++ library not build with recording support");
    return false;
    #endif
}

bool TfliteMicroModel::is_recorder_enabled() const
{
    #if TFLITE_MICRO_RECORDER_ENABLED
    return TfliteMicroRecorder::is_enabled();
    #else 
    return false;
    #endif
}

bool TfliteMicroModel::enable_tensor_recorder()
{
    #if TFLITE_MICRO_RECORDER_ENABLED
    if(is_loaded())
    {
        MLTK_ERROR("Model already loaded");
        return false;
    }
    TfliteMicroRecorder::set_tensor_data_recording_enabled(true);
    return true;
    #else
    MLTK_ERROR("C++ library not build with recording support");
    return false;
    #endif
}

bool TfliteMicroModel::is_tensor_recorder_enabled() const
{
    #if TFLITE_MICRO_RECORDER_ENABLED
    return TfliteMicroRecorder::is_tensor_data_recording_enabled();
    #else 
    return false;
    #endif
}

bool TfliteMicroModel::recorded_data(const uint8_t** buffer_ptr, uint32_t* length_ptr) const
{
    #ifdef TFLITE_MICRO_RECORDER_ENABLED
    return TfliteMicroRecorder::get_recorded_data(buffer_ptr, length_ptr);
    #else 
    return false;
    #endif
}

void TfliteMicroModel::set_processing_callback(void (*callback)(void*), void *arg)
{
    _processing_callback = callback;
    _processing_callback_arg = arg;

    if(is_loaded())
    {
        TfliteMicroModelHelper::set_processing_callback(
            tflite_context(), 
            this->_processing_callback, 
            this->_processing_callback_arg
        );
    }
}

const void* TfliteMicroModel::find_metadata(const char* tag, uint32_t* length) const
{
    return TfliteMicroModelHelper::get_metadata_from_tflite_flatbuffer(this->_flatbuffer, tag, length);
}

bool TfliteMicroModel::load_model_parameters(const void* flatbuffer)
{
    flatbuffer = (flatbuffer == nullptr) ? this->_flatbuffer : flatbuffer;

    if(TfliteModelParameters::load_from_tflite_flatbuffer(flatbuffer, this->parameters))
    {
        _model_details.load_parameters(&this->parameters);
        return true;
    }
    else
    {
        return false;
    }
}

bool TfliteMicroModel::load_interpreter(
    const void* flatbuffer, 
    const tflite::MicroOpResolver* op_resolver,
    uint8_t* buffers[],
    const int32_t buffer_sizes[],
    int32_t buffer_count
)
{
    TfliteMicroContext* mltk_tflm_context = nullptr;
    tflite::MicroAllocator *allocator = nullptr;
    auto tflite_model = tflite::GetModel(flatbuffer);

    // Register the accelerator that was built with this application
    auto accelerator = mltk_tflite_micro_get_registered_accelerator();
    if(accelerator != nullptr)
    {
        // Initialize the accelerator
        if(!accelerator->init())
        {
            MLTK_ERROR("Failed to initialize the accelerator");
            return false;
        }
    }

    if(accelerator != nullptr)
    {
        // Attempt to create a memory allocator for this accelerator.
        // This API will return false if there was an error creating the allocator.
        // The "allocator" argument will be NULL if no allocator was created 
        // (i.e. the model was not compile with a memory plan)
        if(!accelerator->create_allocator(
            flatbuffer,
            buffers, 
            buffer_sizes, 
            buffer_count, 
            &allocator
        ))
        {
            MLTK_ERROR("Failed to create accelerator memory allocator");
            accelerator->deinit(&_interpreter->context_);
            return false;
        }
    }

    if(allocator == nullptr)
    {
        #ifdef MLTK_TFLITE_OFFLINE_MEMORY_PLANNING_REQUIRED
        MLTK_ERROR("Offline memory planning is required for this build");
        unload();
        return false;
        #else
        allocator = tflite::MicroAllocator::Create(
            buffers[0], 
            buffer_sizes[0]
        );
        #endif // MLTK_TFLITE_OFFLINE_MEMORY_PLANNING_REQUIRED
    }

    _interpreter = new(_interpreter_buffer)tflite::MicroInterpreter(
        tflite_model,
        *op_resolver,
        allocator
    );

    auto context = &_interpreter->context_;

    mltk_tflm_context = (accelerator != nullptr) ? 
        accelerator->create_context(context) :
        TfliteMicroContext::create(context);

    if(!mltk_tflm_context->init(
        flatbuffer,
        context, 
        accelerator,
        allocator
    ))
    {
        MLTK_ERROR("Failed to init context");
        unload();
        return false;
    }

    TfliteMicroModelHelper::set_active_tflite_context(tflite_context());

    MLTK_RECORD_START();

    bool retval = true;
    if(_interpreter->AllocateTensors() == kTfLiteOk)
    {
        _ops_resolver = op_resolver;
        _flatbuffer = flatbuffer;

        if(!mltk_tflm_context->load(context))
        {
            MLTK_ERROR("Failed to load context");
            unload();
            return false;
        }
    }
    else 
    {
        unload();
        retval = false;
    }

    TfliteMicroModelHelper::set_active_tflite_context(nullptr);
    // Unsupported kernel message should only be printed the first time we attempt to load the model.
    // Need to all TfliteMicroModel.load() to print them again.
    TfliteMicroKernelMessages::set_enabled(false);

    return retval;
}

/**
 * Do a binary search to find the optimal runtime memory size
*/
bool TfliteMicroModel::find_optimal_buffer_size(
    const void* flatbuffer,
    const tflite::MicroOpResolver* op_resolver,
    uint8_t* buffers[],
    int32_t buffer_sizes[],
    int32_t buffer_count,
    int32_t &optimal_buffer_size
)
{
    #ifdef __arm__
    int upper_limit = __HEAP_SIZE - 8*1024;
    #else
    int upper_limit = SRAM_SIZE;
    #endif
    int lower_limit = 2048;
    int last_working_buffer_size = -1;
    const bool saved_profiler_enabled = TfliteMicroProfiler::is_enabled();
    const bool saved_recording_enabled = TfliteMicroRecorder::is_enabled();
    const auto saved_log_level = get_logger().level();
    TfliteMicroProfiler::set_enabled(false);
    TfliteMicroRecorder::set_enabled(false);
    get_logger().level(logging::Error);

    MLTK_INFO("Searching for optimal runtime memory size ...");


    // Try to get the optimal buffer size to within 128 bytes
    while((upper_limit - lower_limit) > 128)
    {
        // Get the midpoint between the upper and lower limits
        int buffer_size = (upper_limit + lower_limit) / 2;
        buffer_size = ((buffer_size + 16 - 1) / 16) * 16; // align to 16-bytes

        // Allocate the buffer
        uint8_t* buffer = static_cast<uint8_t*>(malloc(buffer_size));
        if(buffer == nullptr)
        {
            // If we failed to malloc, then we don't have enough heap memory
            // So decrease the upper limit by 8k and try again
            // (This should only happen when we first start this algorithm)
            upper_limit -= 8*1024;
            continue;
        }

        buffer_sizes[0] = buffer_size;
        buffers[0] = buffer;

        // Try to load the model with the new buffer
        if(load_interpreter(
            flatbuffer, 
            op_resolver, 
            buffers, 
            buffer_sizes,
            buffer_count
        ))
        {
            // The model was successfully loaded
            // Save the buffer size
            last_working_buffer_size = buffer_size;

            // And set the upper limit to the working buffer size
            // (the goal is to find the smallest possible buffer size)
            upper_limit = buffer_size;

            // Also unload the interpreter so we can load it again
            unload();
        }
        else
        {
            // Immediately break out of the loop
            // if the model has unknown layers
            if(TfliteMicroKernelMessages::unknown_layers_detected())
            {
                break;
            }

            // Otherwise, the buffer size is too small,
            // So the new lower limit is the buffer size+1
            lower_limit = buffer_size+1;
        }

        // Free the buffer for the next iteration
        free(buffer);
    }

    TfliteMicroProfiler::set_enabled(saved_profiler_enabled);
    TfliteMicroRecorder::set_enabled(saved_recording_enabled);
    get_logger().level(saved_log_level);

    if(last_working_buffer_size == -1)
    {
        optimal_buffer_size = 0;
        // Return false if we failed to find a working buffer size
        return false;
    }

    // Add some additional memory for any padding or invoking the context.GetTensor() APIs.
    last_working_buffer_size += 256;

    MLTK_INFO("Determined optimal runtime memory size to be %d", last_working_buffer_size);

    // Otherwise, we found a good buffer size
    // So return success
    optimal_buffer_size = last_working_buffer_size;
    return true;
}



/*************************************************************************************************
 * The following code is used to account for the overhead 64-bit builds add to the runtime memory size.
 * Recall that embedded builds use 32-bit.
 * The following reduces the runtime memory size to convert from 64-bit pointers to 32-bit pointers.
 * The values below were experimentally found
 */
int adjust_required_tensor_arena_bytes_from_64bit_to_32bit(
    const void* flatbuffer,
    tflite::MicroInterpreter* interpreter,
    int tensor_arena_size
)
{
    auto tflite_model = tflite::GetModel(flatbuffer);
    const auto& subgraph = *tflite_model->subgraphs()->Get(0);
    const int tensor_count = subgraph.tensors()->size();
    const int input_count = subgraph.inputs()->size();
    const int output_count = subgraph.outputs()->size();
    const int layer_count = subgraph.operators()->size();
    const int scratch_buffer_count = interpreter->allocator_.scratch_buffer_request_count_;
    int quantize_count = 0;
    int additional_pointer_count = 0;

    for(int i = 0; i < input_count; ++i)
    {
        int index = subgraph.inputs()->Get(i);
        auto tensor = subgraph.tensors()->Get(index);
        if(tensor->quantization() != nullptr)
        {
            ++quantize_count;
        }
    }
    for(int i = 0; i < output_count; ++i)
    {
        int index = subgraph.outputs()->Get(i);
        auto tensor = subgraph.tensors()->Get(index);
        if(tensor->quantization() != nullptr)
        {
            ++quantize_count;
        }
    }

    // Account for the pointers defined in the various MVP kernel OpData structs
    auto accelerator = mltk_tflite_micro_get_registered_accelerator();
    if(accelerator != nullptr && strcmp(accelerator->name(), "mvp") == 0)
    {
        const auto& operators = *subgraph.operators();
        const auto& opcodes = *tflite_model->operator_codes();
        for(const auto op : operators)
        {
            const auto& opcode = opcodes[op->opcode_index()];

            if(opcode->builtin_code() == tflite::BuiltinOperator_ADD)
            {
                additional_pointer_count += 3;
            }
            else if(opcode->builtin_code() == tflite::BuiltinOperator_CONV_2D)
            {
                additional_pointer_count += 7;
            }
            else if(opcode->builtin_code() == tflite::BuiltinOperator_DEPTHWISE_CONV_2D)
            {
                additional_pointer_count += 7;
            }
            else if(opcode->builtin_code() == tflite::BuiltinOperator_FULLY_CONNECTED)
            {
                additional_pointer_count += 5;
            }
            else if(opcode->builtin_code() == tflite::BuiltinOperator_AVERAGE_POOL_2D)
            {
                additional_pointer_count += 2;
            }
            else if(opcode->builtin_code() == tflite::BuiltinOperator_MAX_POOL_2D)
            {
                additional_pointer_count += 2;
            }
            else if(opcode->builtin_code() == tflite::BuiltinOperator_TRANSPOSE_CONV)
            {
                additional_pointer_count += 8;
            }
        }
    }

    const uint32_t overhead_64bit = \
        56*2 + /* SimpleMemoryAllocator x 2 */ \
        72 + /* GreedyMemoryPlanner */ \
        64 + /* MicroAllocator */ \
        16 + /* MicroBuiltinDataAllocator */ \
        16 + /* SubgraphAllocations */ \
        24*tensor_count + /* TfLiteEvalTensor*/ \
        64*layer_count  + /* NodeAndRegistration*/ \
        8*scratch_buffer_count + /* ScratchBufferHandle*/ \
        (8+64)*input_count + /* input TfLiteTensor* + TfLiteTensor */ \
        (8+64)*output_count + /* output TfLiteTensor* + TfLiteTensor */ \
        24 * quantize_count + /*  TfLiteAffineQuantization */ \
        8 * additional_pointer_count;

    const uint32_t overhead_32bit = \
        28*2 + /* SimpleMemoryAllocator x 2 */ \
        44 + /* GreedyMemoryPlanner */ \
        32 + /* MicroAllocator */ \
        8 + /* MicroBuiltinDataAllocator */ \
        8 + /* SubgraphAllocations */ \
        12*tensor_count + /* TfLiteEvalTensor*/ \
        32*layer_count  + /* NodeAndRegistration*/ \
        4*scratch_buffer_count + /* ScratchBufferHandle*/ \
        (4+32)*input_count + /* input TfLiteTensor* + TfLiteTensor */ \
        (4+32)*output_count + /* output TfLiteTensor* + TfLiteTensor */ \
        12 * quantize_count + /*  TfLiteAffineQuantization */ \
        4 * additional_pointer_count;

    tensor_arena_size -= overhead_64bit;
    tensor_arena_size += overhead_32bit;

    return tensor_arena_size;
}

} // namespace mltk
