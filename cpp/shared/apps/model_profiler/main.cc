#include <cstdio>

#include "cpputils/string.hpp"
#include "sl_system_init.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "mltk_tflite_micro_helper.hpp"


// These are defined by the build scripts
// which converts the specified .tflite to a C array
extern "C" const uint8_t sl_tflite_model_array[];
extern "C" const uint32_t sl_tflite_model_len;


using namespace mltk;



static bool load_model(TfliteMicroModel &model, logging::Logger& logger);
static void print_recorded_data(TfliteMicroModel &model, logging::Logger& logger);


tflite::AllOpsResolver op_resolver;



extern "C" int main(void)
{
    TfliteMicroModel model;
    
    sl_system_init();

    auto& logger = get_logger();
    logger.flags(logging::Newline);

    logger.info("Starting Model Profiler");

    if(!load_model(model, logger))
    {
        logger.error("Error while loading model");
        return -1;
    }

    model.print_summary(&logger);

    auto profiler = model.profiler();
    profiling::print_metrics(profiler, &logger);

    if(!model.invoke())
    {
        logger.error("Error while running inference");
        return -1;
    }

    profiling::print_stats(profiler, &logger);
    print_recorded_data(model, logger);
    
    logger.info("done");

    return 0;
}


static bool load_model(TfliteMicroModel &model, logging::Logger& logger)
{
    const uint8_t* tflite_flatbuffer;
    uint32_t tflite_flatbuffer_length;

    // Register the accelerator if the TFLM lib was built with one
    mltk_tflite_micro_register_accelerator();

    // First check if a new .tflite was programmed to the end of flash
    // (This will happen when this app is executed from the command-line: "mltk profiler my_model --device")
    if(!get_tflite_flatbuffer_from_end_of_flash(&tflite_flatbuffer, &tflite_flatbuffer_length))
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
    model.enable_recorder();
#endif

    logger.info("Loading model");

    // Attempt to load the model using the arena size specified in the .tflite
    if(!model.load(tflite_flatbuffer, op_resolver))
    {
        logger.info("Failed to load model");
        return false;
    }

    return true;
}

static void print_recorded_data(TfliteMicroModel &model, logging::Logger& logger)
{
#ifdef TFLITE_MICRO_RECORDER_ENABLED
    logger.info("Recording results:");
    auto& recorded_data = model.recorded_data();
    int layer_idx = 0;
    for(auto& layer : recorded_data)
    {
        logger.info("Layer %d:", layer_idx);
        logger.info("  Input sizes (bytes):");
        for(auto& input : layer.inputs)
        {
            logger.info("    %d", input.length);
        }
        logger.info("  Output sizes (bytes):", layer_idx);
        for(auto& output : layer.outputs)
        {
            logger.info("    %d", output.length);
        }
        ++layer_idx;
    }
    recorded_data.clear();
#endif
}
