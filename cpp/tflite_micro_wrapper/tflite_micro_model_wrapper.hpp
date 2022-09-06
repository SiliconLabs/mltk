#include <string>
#include <map>
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
        bool force_buffer_overlap,
        int runtime_memory_size
    );

    bool invoke() const;
    py::dict get_details() const;
    py::array get_input(int index);
    py::array get_output(int index);
    py::list get_profiling_results() const;
    py::list get_recorded_data();

private:
    const void* _accelerator_wrapper;
    std::string _flatbuffer_data;
    std::string _runtime_memory;
};


} // namespace mltk