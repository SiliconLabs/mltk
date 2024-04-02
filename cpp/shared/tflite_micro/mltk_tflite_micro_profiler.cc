#ifndef MLTK_DLL_IMPORT
#include <cstdlib>
#include <new>

#include "mltk_tflite_micro_profiler.hpp"
#include "mltk_tflite_micro_model_helper.hpp"


namespace mltk 
{

TfliteMicroProfiler& TfliteMicroProfiler::instance()
{
    static uint8_t instance_buffer[sizeof(TfliteMicroProfiler)];
    static TfliteMicroProfiler* instance_ptr = nullptr;

    if(instance_ptr == nullptr)
    {
        instance_ptr = new(instance_buffer)TfliteMicroProfiler();
    }

    return *instance_ptr;
}


void TfliteMicroProfiler::set_enabled(bool enabled)
{
    auto& self = instance();
    self._enabled = enabled;
}


bool TfliteMicroProfiler::is_enabled()
{
    auto& self = instance();
    return self._enabled;
}


void TfliteMicroProfiler::init(int count)
{
    auto& self = instance();

    if(!self._enabled)
    {
        return;
    }

    if(!profiling::register_profiler("Inference", self._inference_profiler))
    {
        return;
    }
    self._inference_profiler->flags(profiling::Flag::ReportTotalChildrenCycles|profiling::Flag::ReportsFreeRunningCpuCycles);

    self._layer_profilers = static_cast<profiling::Profiler**>(malloc(sizeof(profiling::Profiler*)*count));
    if(self._layer_profilers == nullptr)
    {
        profiling::unregister(self._inference_profiler);
        self._inference_profiler = nullptr;
    }
}


void TfliteMicroProfiler::deinit()
{
    auto& self = instance();

    if(self._inference_profiler != nullptr)
    {
        // Unregister the inference profiler and all its children profilers
        profiling::unregister(self._inference_profiler);
        self._inference_profiler = nullptr;
    }
    if(self._layer_profilers != nullptr)
    {
        free(self._layer_profilers);
        self._layer_profilers = nullptr;
    }
}


void TfliteMicroProfiler::register_profiler(
    TfLiteContext* context,
    int layer_index,
    tflite::BuiltinOperator opcode,
    const tflite::NodeAndRegistration& node_and_registration
)
{
    auto& self = instance();

    if(self._inference_profiler == nullptr)
    {
        return;
    }

    profiling::Profiler* profiler;
    profiling::register_profiler(
        TfliteMicroModelHelper::create_layer_name(layer_index, opcode), 
        profiler, 
        self._inference_profiler
    );

    if(profiler == nullptr)
    {
        return;
    }

    profiler->flags().set(profiling::Flag::TimeMeasuredBetweenStartAndStop);
    self._layer_profilers[layer_index] = profiler;
    calculate_op_metrics(context, node_and_registration, profiler->metrics());
}


void TfliteMicroProfiler::start(int loop_index)
{
    auto& self = instance();

    if(self._inference_profiler != nullptr && loop_index <= 0)
    {
        self._inference_profiler->start();
    }

    auto accelerator = mltk_tflite_micro_get_registered_accelerator();
    if(accelerator != nullptr)
    {
        accelerator->start_profiler(loop_index);
    }
}


void TfliteMicroProfiler::stop(int loop_index)
{
    auto& self = instance();

    auto accelerator = mltk_tflite_micro_get_registered_accelerator();
    if(accelerator != nullptr)
    {
        accelerator->stop_profiler(loop_index);
    }

    if(self._inference_profiler != nullptr && loop_index <= 0)
    {
        self._inference_profiler->stop();
    }
}


void TfliteMicroProfiler::start_layer(int index)
{
    auto& self = instance();

    if(self._layer_profilers != nullptr)
    {
        self._layer_profilers[index]->start();
    }

    auto accelerator = mltk_tflite_micro_get_registered_accelerator();
    if(accelerator != nullptr)
    {
        auto layer_profiler = (self._layer_profilers != nullptr) ? self._layer_profilers[index] : nullptr;
        accelerator->start_op_profiler(index, layer_profiler);
    }
}


void TfliteMicroProfiler::stop_layer(int index)
{
    auto& self = instance();

    if(self._layer_profilers != nullptr)
    {
        self._layer_profilers[index]->stop();
    }

    auto accelerator = mltk_tflite_micro_get_registered_accelerator();
    if(accelerator != nullptr)
    {
        auto layer_profiler = (self._layer_profilers != nullptr) ? self._layer_profilers[index] : nullptr;
        accelerator->stop_op_profiler(index, layer_profiler);
    }
}

} // namespace mltk 


#endif // #ifndef MLTK_DLL_IMPORT