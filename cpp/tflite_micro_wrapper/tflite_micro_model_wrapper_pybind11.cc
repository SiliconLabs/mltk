#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "tflite_micro_model_wrapper.hpp"


namespace py = pybind11;
using namespace mltk;



void init_tflite_micro_model(py::module &m)
{
    py::class_<TfliteMicroModelWrapper>(m, "TfliteMicroModelWrapper")
    .def(py::init<>())
    .def("load", &TfliteMicroModelWrapper::load)
    .def("get_details", &TfliteMicroModelWrapper::get_details)
    .def("get_input_size", &TfliteMicroModelWrapper::input_size)
    .def("get_input", &TfliteMicroModelWrapper::get_input)
    .def("get_output_size", &TfliteMicroModelWrapper::output_size)
    .def("get_output", &TfliteMicroModelWrapper::get_output)
    .def("invoke", &TfliteMicroModelWrapper::invoke)
    .def("is_profiler_enabled", &TfliteMicroModelWrapper::profiler_is_enabled)
    .def("get_profiling_results", &TfliteMicroModelWrapper::get_profiling_results)
    .def("is_tensor_recorder_enabled", &TfliteMicroModelWrapper::is_tensor_recorder_enabled)
    .def("get_recorded_data", &TfliteMicroModelWrapper::get_recorded_data)
    ;
}