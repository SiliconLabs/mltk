#include <cstdio>

#include "cpputils/string.hpp"
#include "sl_system_init.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "mltk_tflite_micro_helper.hpp"
#include "mltk_tflite_micro_accelerator_recorder.hpp"

#ifndef __arm__
// CLI parsing only supported on Windows/Linux
#include "cli_opts.hpp"
#endif



using namespace mltk;




#ifdef TFLITE_MICRO_RECORDER_ENABLED
static void dump_recorded_data(TfliteMicroModel &model, logging::Logger& logger);
#endif

static bool load_model(
    TfliteMicroModel &model,
    logging::Logger& logger,
    const uint8_t* tflite_input_flatbuffer,
    uint32_t tflite_input_flatbuffer_len
);





// These are defined by the build scripts
// which converts the specified .tflite to a C array
extern "C" const uint8_t sl_tflite_model_array[];
extern "C" const uint32_t sl_tflite_model_len;


#if __has_include("mltk_model_profiler_generated_model_op_resolver.hpp")
    // Check if mltk_model_profiler_generated_model_op_resolver.hpp was automatically generated.
    // If so, the header contains only the TFLM kernels that are required by the given model.
    // This can *greatly* reduce the app's flash space.
    // To enable this feature, add the CMake variable: MODEL_PROFILER_GENERATE_OP_RESOLVER_HEADER
    #include "mltk_model_profiler_generated_model_op_resolver.hpp"
    MyOpResolver op_resolver;
#else
    // Otherwise, just build all available TFLM kernels into the app.
    // While this consume *a lot* more flash space,
    // it allows for dynamically loading .tflite models.
    tflite::AllOpsResolver op_resolver;
#endif


#if defined(MODEL_PROFILER_RUNTIME_MEMORY_SIZE) && MODEL_PROFILER_RUNTIME_MEMORY_SIZE > 0
#   ifndef MODEL_PROFILER_RUNTIME_MEMORY_SECTION
#     define MODEL_PROFILER_RUNTIME_MEMORY_SECTION ".bss"
#   endif
    // If the runtime memory size (i.e. tensor arena) was defined a compile-time
    // then define a global buffer
    const int runtime_memory_size = MODEL_PROFILER_RUNTIME_MEMORY_SIZE;
    uint8_t __attribute__((section(MODEL_PROFILER_RUNTIME_MEMORY_SECTION))) runtime_memory_buffer[MODEL_PROFILER_RUNTIME_MEMORY_SIZE];
#else
    // Otherwise, we dynamically find the optimal runtime size
    // and allocate from the heap
    const int runtime_memory_size = -1;
    uint8_t* runtime_memory_buffer = nullptr;
#endif


extern "C" int main(void)
{
    TfliteMicroModel model;

    sl_system_init();

    auto& logger = get_logger();
    logger.flags(logging::Newline);

    logger.info("Starting Model Profiler");

#ifndef __arm__
    // If this is a Windows/Linux build
    // Parse the CLI options
    parse_cli_opts();

    // If no model path was given on the command-line
    // then use the default model built into the app
    if(cli_opts.model_flatbuffer == nullptr)
    {
        logger.info("No model path given. Using default built into application");
        cli_opts.model_flatbuffer = sl_tflite_model_array;
        cli_opts.model_flatbuffer_len = sl_tflite_model_len;
    }

    if(!load_model(model, logger, cli_opts.model_flatbuffer, cli_opts.model_flatbuffer_len))
    {
        logger.error("Error while loading model");
        return -1;
    }
#else
    if(!load_model(model, logger, nullptr, 0))
    {
        logger.error("Error while loading model");
        return -1;
    }
#endif // ifndef __arm__

    model.print_summary(&logger);

    auto profiler = model.profiler();
    profiling::print_metrics(profiler, &logger);

    if(!model.invoke())
    {
        logger.error("Error while running inference");
        return -1;
    }

    profiling::print_stats(profiler, &logger);
#ifdef TFLITE_MICRO_RECORDER_ENABLED
    dump_recorded_data(model, logger);
#endif

    logger.info("done");

    return 0;
}


static bool load_model(
    TfliteMicroModel &model,
    logging::Logger& logger,
    const uint8_t* tflite_input_flatbuffer,
    uint32_t tflite_input_flatbuffer_len
)
{
    const uint8_t* tflite_flatbuffer;
    uint32_t tflite_flatbuffer_length;

    // Register the accelerator if the TFLM lib was built with one
    mltk_tflite_micro_register_accelerator();

    // If a valid model flatbuffer is inputted use that
    if (tflite_input_flatbuffer != nullptr && tflite_input_flatbuffer_len > 0)
    {
        logger.info("Loading provided model");
        tflite_flatbuffer = tflite_input_flatbuffer;
        tflite_flatbuffer_length = tflite_input_flatbuffer_len;
    }
    // First check if a new .tflite was programmed to the end of flash
    // (This will happen when this app is executed from the command-line: "mltk profiler my_model --device")
    else if (!get_tflite_flatbuffer_from_end_of_flash(&tflite_flatbuffer, &tflite_flatbuffer_length))
    {
         // If no .tflite was programmed, then just use the default model
        printf("Using default model built into application\n");
        tflite_flatbuffer = sl_tflite_model_array;
        tflite_flatbuffer_length = sl_tflite_model_len;
    }

#ifdef MLTK_RUN_MODEL_FROM_RAM
    uint8_t* tflite_ram_buffer = (uint8_t*)malloc(tflite_flatbuffer_length);
    if(tflite_ram_buffer == nullptr)
    {
        logger.error("Cannot load .tflite into RAM. Failed to allocate %d bytes for buffer", tflite_flatbuffer_length);
        return -1;
    }
    memcpy(tflite_ram_buffer, tflite_flatbuffer, tflite_flatbuffer_length);
    tflite_flatbuffer = tflite_ram_buffer;
    logger.info("Loaded .tflite into RAM");
#endif // MLTK_RUN_MODEL_FROM_RAM


    model.enable_profiler();
#ifdef TFLITE_MICRO_RECORDER_ENABLED
    model.enable_tensor_recorder();
#endif
#ifdef TFLITE_MICRO_ACCELERATOR_RECORDER_ENABLED
   TfliteMicroAcceleratorRecorder::instance().set_program_recording_enabled();
   TfliteMicroAcceleratorRecorder::instance().set_data_recording_enabled();
#endif

    logger.info("Loading model");
    if(!model.load(
        tflite_flatbuffer,
        op_resolver,
        runtime_memory_buffer,
        runtime_memory_size
    ))
    {
        logger.info("Failed to load model");
        return false;
    }

    return true;
}

#ifdef TFLITE_MICRO_RECORDER_ENABLED
#include "mltk_tflite_micro_recorder.hpp"

static void dump_recorded_data(TfliteMicroModel &model, logging::Logger& logger)
{
    const uint8_t* buffer;
    uint32_t buffer_length;
    msgpack_object_t* root_obj;

    if(!model.recorded_data(&buffer, &buffer_length))
    {
         logger.error("No recorded data available");
        return;
    }
    if(msgpack_deserialize_with_buffer(&root_obj, buffer, buffer_length, MSGPACK_FLAGS_NONE) != 0)
    {
         logger.error("Failed to de-serialize recorded data");
        return;
    }

    struct Context {
        logging::Logger& logger;
        int n_values = 0;
        int current_index = 0;
        int layer_index = 0;

        Context(logging::Logger& logger): logger(logger){}
    };

    auto msgpack_iterator = [](const msgpack_object_t *key, const msgpack_object_t *value, void *arg) -> int
    {
        auto& context = *(Context*)arg;

        if(context.current_index == context.n_values)
        {
            if(key == nullptr)
            {
                context.logger.info("Layer %d:", context.layer_index);
                context.layer_index += 1;
                return 0;
            }

            char key_str[256];
            context.current_index = 0;
            context.logger.info("  %s:", msgpack_to_str(key, key_str, sizeof(key_str)));

            if(MSGPACK_IS_ARRAY(value))
            {
                context.n_values = MSGPACK_ARRAY_LENGTH(value);
            }
            else if(MSGPACK_IS_DICT(value))
            {
                context.n_values = MSGPACK_DICT_LENGTH(value);
            }
            else
            {
                return -1;
            }
        }
        else if(key == nullptr)
        {
            context.logger.info("    %d: %d elements", context.current_index, MSGPACK_BIN_LENGTH(value));
            context.current_index += 1;
        }
        else
        {
            char key_str[64];
            char val_str[64];
            context.logger.info("    %s: %s", msgpack_to_str(key, key_str, sizeof(key_str)), msgpack_to_str(value, val_str, sizeof(val_str)));
            context.current_index += 1;
        }

        return 0;
    };



    Context context(logger);

    logger.info("Recording results (%d layers):", MSGPACK_ARRAY_LENGTH(root_obj));
    if(msgpack_foreach(root_obj, msgpack_iterator, &context, 10) != 0)
    {
        logger.error("Failed to process recorded data");
    }

    msgpack_free_objects(root_obj);
}
#endif