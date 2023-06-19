#include <exception>

#include "all_ops_resolver.h"
#include "tflite_micro_model_wrapper.hpp"
#include "tflite_micro_accelerator_wrapper.hpp"
#include "mltk_tflite_micro_helper.hpp"
#include "pybind11_helper.hpp"


extern bool mltk_tflm_force_buffer_overlap;

namespace mltk
{

static tflite::AllOpsResolver reference_ops_resolver;

/*************************************************************************************************/
TfliteMicroModelWrapper::~TfliteMicroModelWrapper()
{
    unload();
    mltk_tflite_micro_set_accelerator(nullptr);
}

/*************************************************************************************************/
bool TfliteMicroModelWrapper::load(
    const std::string& flatbuffer_data,
    void* accelerator,
    bool enable_profiler,
    bool enable_tensor_recorder,
    bool force_buffer_overlap,
    int runtime_memory_size
)
{
    get_logger().debug("Loading model ...");

    tflite::MicroOpResolver *op_resolver;
    this->_flatbuffer_data = flatbuffer_data;
    this->_accelerator_wrapper = accelerator;

    if(enable_profiler)
    {
        this->enable_profiler();
    }
    if(enable_tensor_recorder)
    {
        this->enable_tensor_recorder();
    }

    // If no accelerator is provided,
    // then just use the reference kernels
    if(accelerator == nullptr)
    {
        op_resolver = &reference_ops_resolver;
    }
    else
    {
        auto acc = (TfliteMicroAcceleratorWrapper*)accelerator;
        op_resolver = acc->load();
    }

    mltk_tflm_force_buffer_overlap = force_buffer_overlap;

    uint8_t* runtime_buffer = nullptr;
    if(runtime_memory_size > 0)
    {
        _runtime_memory.reserve(runtime_memory_size);
        runtime_buffer = (uint8_t*)_runtime_memory.c_str();
    }

    bool retval = TfliteMicroModel::load(
        this->_flatbuffer_data.c_str(),
        *op_resolver,
        runtime_buffer,
        runtime_memory_size
    );

    mltk_tflm_force_buffer_overlap = false;

    return retval;
}

/*************************************************************************************************/
bool TfliteMicroModelWrapper::invoke() const
{
    if(this->_accelerator_wrapper != nullptr)
    {
        // Technically, this should not be required as the accelerator should have already
        // been set by the accelerator wrapper when calling mltk_tflite_micro_register_accelerator() in the load()
        // However, for some reason the pointer is being cleared on Linux in some instances.
        // So, as a failsafe we set the accelerator pointer again before invoking the simulator.
        auto accelerator_wrapper = (const TfliteMicroAcceleratorWrapper*)this->_accelerator_wrapper;
        mltk_tflite_micro_set_accelerator(accelerator_wrapper->accelerator);
    }

    return TfliteMicroModel::invoke();
}

/*************************************************************************************************/
py::dict TfliteMicroModelWrapper::get_details() const
{
    auto& details = this->details();
    py::dict details_dict;

    details_dict["name"] = details.name();
    details_dict["version"] = details.version();
    details_dict["date"] = details.date();
    details_dict["description"] = details.description();
    details_dict["hash"] = details.hash();
    details_dict["accelerator"] = details.accelerator();
    details_dict["runtime_memory_size"] = details.runtime_memory_size();
    py::list classes;
    for(auto c : details.classes())
    {
        classes.append(c);
    }
    details_dict["classes"] = classes;

    // These are used internally for debugging
    details_dict["tflite_buffer_addr"] = (uint32_t)((uintptr_t)this->_flatbuffer_data.c_str());
    details_dict["tflite_buffer_Length"] = this->_flatbuffer_data.length();


    return details_dict;
}

/*************************************************************************************************/
py::array TfliteMicroModelWrapper::get_input(int index)
{
    auto tensor = this->input(index);

    if(tensor == nullptr)
    {
        throw std::out_of_range("Invalid input tensor index");
    }

    return tflite_tensor_to_array(*tensor);
}

/*************************************************************************************************/
py::array TfliteMicroModelWrapper::get_output(int index)
{
    auto tensor = this->output(index);

    if(tensor == nullptr)
    {
        throw std::out_of_range("Invalid output tensor index");
    }

    return tflite_tensor_to_array(*tensor);
}

/*************************************************************************************************/
py::list TfliteMicroModelWrapper::get_profiling_results() const
{
    py::list results;

    const auto model_profiler = this->profiler();
    if(model_profiler == nullptr)
    {
        return results;
    }

    cpputils::TypedList<profiling::Profiler*> profiler_list;
    profiling::get_all(model_profiler->name(), profiler_list);
    for(const auto profiler : profiler_list)
    {
        // Only return model layer profilers which have name like:
        // Op3-Conv2d
        if(strncmp(profiler->name(), "Op", 2) != 0)
        {
            continue;
        }

        py::dict result;
        const auto& stats = profiler->stats();
        const auto& metrics = profiler->metrics();
        result["name"] = profiler->name();
        result["macs"] = metrics.macs;
        result["ops"] = metrics.ops;
        result["accelerator_cycles"] = stats.accelerator_cycles;
        result["cpu_cycles"] = 0; // stats.cpu_cycles;  CPU cycles and time are not currently measured in the simulator
        result["time"] = 0; // (float)stats.time_us / 1e6;
        for(auto item_it = profiler->custom_stats.items(); item_it != profiler->custom_stats.enditems(); ++item_it)
        {
            const auto item = item_it.current;
            result[item->key] = *item->value;
        }

        results.append(result);
    }

    return results;
}

/*************************************************************************************************/
py::bytes TfliteMicroModelWrapper::get_recorded_data()
{
    const uint8_t* data;
    uint32_t length;

    if(this->recorded_data(&data, &length))
    {
        std::string buf((const char*)data, length);
        return py::bytes(buf);
    }

    return py::none();
}





} // namespace mltk