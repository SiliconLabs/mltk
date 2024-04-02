#include <string>
#include <map>
#include <vector>
#include <tuple>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>


#include "tflite_micro_model/tflite_micro_model.hpp"


namespace py = pybind11;


namespace mltk
{

class TfliteMicroModelWrapper : public TfliteMicroModel
{
public:
    ~TfliteMicroModelWrapper();
    bool load(
        const std::string& flatbuffer_data, 
        void* accelerator,
        bool enable_profiler,
        bool enable_recorder,
        bool enable_tensor_recorder,
        bool force_buffer_overlap,
        const std::vector<int>& runtime_memory_sizes
    );
    void unload();

    bool invoke() const;
    py::dict get_details() const;
    py::array get_input(int index);
    py::array get_output(int index);
    py::list get_profiling_results() const;
    py::object get_recorded_data();
    py::list get_layer_msgs() const;
    void set_layer_callback(std::function<bool(py::dict)> callback);

private:
    const void* _accelerator_wrapper;
    std::string _flatbuffer_data;
    std::vector<std::string*> _runtime_buffers;
    std::function<bool(py::dict)> _layer_callback;
    std::vector<std::string> _layer_msgs;

    static TfLiteStatus layer_callback_handler(
        int index,
        TfLiteContext& context,
        const tflite::NodeAndRegistration& node_and_registration,
        TfLiteStatus invoke_status,
        void* arg
    );
    static void kernel_message_callback(
        const char* msg, 
        void *arg
    );
};


} // namespace mltk