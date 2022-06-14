

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "audio_feature_generator_wrapper.hpp"


namespace py = pybind11;


PYBIND11_MODULE(MODULE_NAME, m) 
{

    /*************************************************************************************************
     * API version number of the wrapper 
     */
    m.def("api_version", []() -> int
    {
        return AUDIO_FEATURE_GENERATOR_API_VERSION;
    });

    /*************************************************************************************************
     * GIT hash of the MLTK repo when the DLL was compiled
     */
    m.def("git_hash", []() -> const char*
    {
        return GIT_HASH;
    });

    py::class_<mltk::AudioFeatureGeneratorWrapper>(m, "AudioFeatureGeneratorWrapper")
    .def(py::init<const py::dict&>())
    .def("process_sample", &mltk::AudioFeatureGeneratorWrapper::process_sample)
    .def("activity_was_detected", &mltk::AudioFeatureGeneratorWrapper::activity_was_detected)
    ;
}