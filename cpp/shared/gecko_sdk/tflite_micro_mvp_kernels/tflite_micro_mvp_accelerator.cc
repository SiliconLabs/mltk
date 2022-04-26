

#include "sl_mvp.h"

#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
#include "sl_mvp_simulator.hpp"
#endif

#include "mltk_tflite_micro_helper.hpp"
#include "mltk_tflite_micro_accelerator_recorder.hpp"


namespace mltk
{

const char* NAME = "MVP";

static unsigned int dma_channel = -1;
static const char* perfcnt_names[] = 
{
    "run",
    "cmd",
    "stall",
    "noop",
    "alu-active",
    "pipe-stall",
    "io-fence-stall",
    "load0-stall",
    "load1-stall",
    "store-stall",
    "bus-stall",
    "load0-ahb-stall",
    "load1-ahb-stall",
    "load0-fence-stall",
    "load1-fence-stall"
};
static int current_loop_index = 0;


/*************************************************************************************************/
static void init_accelerator()
{
#ifdef __arm__
    sli_mvp_init();
#else 
    // NOTE: We do NOT call sli_mvp_config() for non-arm
    // because it must be called from within the simulator
    // As such, the sli_mvp_execute() API calls sli_mvp_config()
#endif
}

/*************************************************************************************************/
static void deinit_accelerator()
{
    sli_mvp_deinit();
}

/*************************************************************************************************/
static int get_profiler_loop_count()
{
    return 7;
}

/*************************************************************************************************/
static void start_profiler(int loop_index)
{
#ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
    current_loop_index = loop_index;
    sli_mvp_perfcnt_conf(0, (sli_mvp_perfcnt_t)(loop_index*2));
    sli_mvp_perfcnt_conf(1, (sli_mvp_perfcnt_t)(loop_index*2+1));
#endif
}

/*************************************************************************************************/
static void stop_profiler(int loop_index)
{
}

/*************************************************************************************************/
static void start_op_profiler(int op_idx, profiling::Profiler* profiler)
{
    sli_mvp_perfcnt_reset_all();
#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
    if(profiler != nullptr) sli_mvp_set_current_profiler(profiler);
#endif
    TFLITE_MICRO_ACCELERATOR_RECORDER_START_LAYER();
}

/*************************************************************************************************/
static void stop_op_profiler(int op_idx, profiling::Profiler* profiler)
{
#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
    sli_mvp_set_current_profiler(nullptr);
#endif

    if(profiler != nullptr)
    {
#ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
        const uint32_t percnt0 = sli_mvp_perfcnt_get(0);
        const uint32_t percnt1 = sli_mvp_perfcnt_get(1);
        if(current_loop_index == 0)
        {
            profiler->stats().accelerator_cycles = percnt0;
        }
        profiler->parent()->increment_custom_stat(perfcnt_names[current_loop_index*2], percnt0);
        profiler->parent()->increment_custom_stat(perfcnt_names[current_loop_index*2+1], percnt1);
        profiler->increment_custom_stat(perfcnt_names[current_loop_index*2], percnt0);
        profiler->increment_custom_stat(perfcnt_names[current_loop_index*2+1], percnt1);
#else
        profiler->stats().accelerator_cycles = sli_mvp_perfcnt_get(0);
#endif 
    }  

    TFLITE_MICRO_ACCELERATOR_RECORDER_END_LAYER();
}



static const TfliteMicroAccelerator accelerator = 
{
    /*name*/NAME,
    /*init*/init_accelerator,
    /*deinit*/deinit_accelerator,
    /*get_profiler_loop_count*/get_profiler_loop_count,
    /*start_profiler*/start_profiler,
    /*stop_profiler*/stop_profiler,
    /*start_op_profiler*/start_op_profiler,
    /*stop_op_profiler*/stop_op_profiler,
#ifdef TFLITE_MICRO_SIMULATOR_ENABLED
    /*set_simulator_memory*/sli_mvp_set_simulator_memory,
    /*invoke_simulator*/sli_mvp_invoke_in_simulator
#endif
};


/*************************************************************************************************/
extern "C" const TfliteMicroAccelerator* mltk_tflite_micro_get_accelerator()
{
    return &accelerator;
}

/*************************************************************************************************/
extern "C" void mltk_tflite_micro_register_accelerator()
{
    mltk_tflite_micro_set_accelerator(&accelerator);
}


} // namespace mltk