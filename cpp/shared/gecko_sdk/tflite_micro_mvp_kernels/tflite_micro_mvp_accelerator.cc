

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

class MvpTfliteMicroAccelerator : public TfliteMicroAccelerator
{
    int current_loop_index = 0;

public:

    const char* name() const override
    {
        return "mvp";
    }

    bool init() override
    {
        return sli_mvp_init() == SL_STATUS_OK;
    }

    void deinit(TfLiteContext *context) override
    {
        sli_mvp_deinit();
    }

    int get_profiler_loop_count() override
    {
        return 7;
    }

    #ifdef TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED
    void start_profiler(int loop_index) override
    {
        current_loop_index = loop_index;
        sli_mvp_perfcnt_conf(0, (sli_mvp_perfcnt_t)(loop_index*2));
        sli_mvp_perfcnt_conf(1, (sli_mvp_perfcnt_t)(loop_index*2+1));
    }
    #endif // TFLITE_MICRO_ACCELERATOR_PROFILER_ENABLED


    void start_op_profiler(int op_idx, profiling::Profiler* profiler) override
    {
        sli_mvp_perfcnt_reset_all();
        #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
        if(profiler != nullptr) sli_mvp_set_current_profiler(profiler);
        #endif
    }

    void stop_op_profiler(int op_idx, profiling::Profiler* profiler) override
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
    }

    #ifdef TFLITE_MICRO_SIMULATOR_ENABLED
    bool set_simulator_memory(const char* region, void* base_address, uint32_t length) override
    {
        return sli_mvp_set_simulator_memory(region, base_address, length);
    }

    bool invoke_simulator(const std::function<bool()>&func) override
    {
        return sli_mvp_invoke_in_simulator(func);
    }
    #endif // TFLITE_MICRO_SIMULATOR_ENABLED
};




static MvpTfliteMicroAccelerator mvp_accelerator;

/*************************************************************************************************/
extern "C" TfliteMicroAccelerator* mltk_tflite_micro_get_accelerator()
{
    return &mvp_accelerator;
}

/*************************************************************************************************/
extern "C" TfliteMicroAccelerator* mltk_tflite_micro_register_accelerator()
{
    return mltk_tflite_micro_set_accelerator(mltk_tflite_micro_get_accelerator());
}


} // namespace mltk