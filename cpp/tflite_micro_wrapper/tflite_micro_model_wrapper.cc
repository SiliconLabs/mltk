#include <exception>

#include "tensorflow/lite/micro/memory_helpers.h"
#include "all_ops_resolver.h"
#include "tflite_micro_model_wrapper.hpp"
#include "tflite_micro_accelerator_wrapper.hpp"
#include "mltk_tflite_micro_helper.hpp"
#include "pybind11_helper.hpp"
#include "mltk_tflite_micro_accelerator_recorder.hpp"


extern bool mltk_tflm_force_buffer_overlap;

namespace mltk
{

static tflite::AllOpsResolver reference_ops_resolver;

/*************************************************************************************************/
TfliteMicroModelWrapper::~TfliteMicroModelWrapper()
{
    unload();
}

/*************************************************************************************************/
bool TfliteMicroModelWrapper::load(
    const std::string& flatbuffer_data,
    void* accelerator,
    bool enable_profiler,
    bool enable_recorder,
    bool enable_tensor_recorder,
    bool force_buffer_overlap,
    const std::vector<int>& runtime_memory_sizes
)
{
    if(runtime_memory_sizes.size() == 0)
    {
        get_logger().error("Must provide runtime memory size(s)");
        return false;
    }

    TfliteMicroKernelMessages::set_flush_callback(kernel_message_callback, this);

    get_logger().debug("Loading model ...");

    // This ensures the accelerator recorded is included in the python wrapper DLL
    TfliteMicroAcceleratorRecorder::instance();

    tflite::MicroOpResolver *op_resolver;
    this->_flatbuffer_data = flatbuffer_data;
    this->_accelerator_wrapper = accelerator;

    if(enable_profiler)
    {
        this->enable_profiler();
    }
    if(enable_recorder)
    {
        this->enable_recorder();
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

    const int runtime_buffer_count = runtime_memory_sizes.size();
    uint8_t* runtime_buffers[runtime_buffer_count];
    int32_t runtime_buffer_sizes[runtime_buffer_count];

    for(int i = 0; i < runtime_buffer_count; ++i)
    {
        runtime_buffer_sizes[i] = runtime_memory_sizes[i];
        if(runtime_buffer_sizes[i] > 0)
        {
            auto buffer = new std::string();
            buffer->reserve(runtime_buffer_sizes[i]);
            _runtime_buffers.push_back(buffer);
            runtime_buffers[i] = (uint8_t*)buffer->c_str();
            memset(runtime_buffers[i], 0, runtime_buffer_sizes[i]);
        }
        else 
        {
            runtime_buffers[i] = nullptr;
        }
    }

    bool retval = TfliteMicroModel::load(
        this->_flatbuffer_data.c_str(),
        op_resolver,
        runtime_buffers,
        runtime_buffer_sizes,
        runtime_buffer_count
    );

    mltk_tflm_force_buffer_overlap = false;

    return retval;
}

/*************************************************************************************************/
void TfliteMicroModelWrapper::unload()
{
    TfliteMicroModel::unload();
    mltk_tflite_micro_set_accelerator(nullptr);
    TfliteMicroKernelMessages::set_flush_callback(nullptr);
    for(auto buffer : _runtime_buffers)
    {
        delete buffer;
    }
    _runtime_buffers.clear();
    _layer_msgs.clear();
    _layer_callback = nullptr;
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
py::object TfliteMicroModelWrapper::get_recorded_data()
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

/*************************************************************************************************/
py::list TfliteMicroModelWrapper::get_layer_msgs() const
{
    py::list msgs;

    for(auto& e : _layer_msgs)
    {
        msgs.append(e);
    }

    return msgs;
}

/*************************************************************************************************/
void TfliteMicroModelWrapper::set_layer_callback(std::function<bool(py::dict)> callback)
{
    if(callback == nullptr)
    {
        TfliteMicroModelHelper::set_layer_callback(tflite_context(), nullptr);
        return;
    }

    _layer_callback = callback;
    TfliteMicroModelHelper::set_layer_callback(
        tflite_context(), 
        TfliteMicroModelWrapper::layer_callback_handler, 
        this
    );
}

/*************************************************************************************************/
TfLiteStatus TfliteMicroModelWrapper::layer_callback_handler(
    int index,
    TfLiteContext& context,
    const tflite::NodeAndRegistration& node_and_registration,
    TfLiteStatus invoke_status,
    void* arg
)
{
    auto& self = *reinterpret_cast<TfliteMicroModelWrapper*>(arg);

    py::dict callback_arg;
    py::list outputs;
    callback_arg["index"] = index;
    callback_arg["outputs"] = outputs;


    for(int i = 0; i < node_and_registration.node.outputs->size; ++i)
    {
        auto output_index = node_and_registration.node.outputs->data[i];
        auto output_tensor = context.GetEvalTensor(&context, output_index);
        if(output_tensor == nullptr)
        {
            get_logger().error("Output tensor-%d is null when it shouldn't be", i);
            return kTfLiteError;
        }

        const auto output_shape = tflite::micro::GetTensorShape(output_tensor);
        const auto output_flat_size = output_shape.FlatSize();
        size_t output_dtype_len;

        tflite::TfLiteTypeSizeOf(output_tensor->type, &output_dtype_len);
        const int output_byte_len = output_flat_size*output_dtype_len;

        std::string buf((const char*)output_tensor->data.raw, output_byte_len);
        outputs.append(py::bytes(buf));
    }

    const bool retval = self._layer_callback(callback_arg);

    return retval ? kTfLiteOk : kTfLiteError;
}


/*************************************************************************************************/
void TfliteMicroModelWrapper::kernel_message_callback(
    const char* msg, 
    void *arg
)
{
    auto& self = *reinterpret_cast<TfliteMicroModelWrapper*>(arg);
    self._layer_msgs.push_back(std::string(msg));
}

} // namespace mltk