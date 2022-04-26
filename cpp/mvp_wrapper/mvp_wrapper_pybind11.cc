

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mvp_tflite_micro_accelerator.hpp"
#include "mltk_tflite_micro_accelerator_recorder.hpp"


namespace py = pybind11;


static mltk::MvpTfliteMicroAcceleratorWrapper mvp_accelerator;
extern "C" void sli_mvp_set_simulator_backend_enabled(bool);

namespace mltk
{
    extern bool mvpv1_calculate_accelerator_cycles_only;
    extern "C" const TfliteMicroAccelerator* mltk_tflite_micro_get_accelerator();
}


PYBIND11_MODULE(MODULE_NAME, m) 
{

    /*************************************************************************************************
     * API version number of the wrapper 
     * 
     * Any other wrappers that link to this wrapper should check
     * this version to ensure compatibility
     */
    m.def("name", []() -> const char*
    {
        auto acc = mltk::mltk_tflite_micro_get_accelerator();
        return acc->name;
    });

    /*************************************************************************************************
     * API version number of the wrapper 
     * 
     * Any other wrappers that link to this wrapper should check
     * this version to ensure compatibility
     */
    m.def("api_version", []() -> int
    {
        return TFLITE_MICRO_API_VERSION;
    });

    /*************************************************************************************************
     * GIT hash of the MLTK repo when the DLL was compiled
     * 
     */
    m.def("git_hash", []() -> const char*
    {
        return MLTK_GIT_HASH;
    });

    /*************************************************************************************************
     * Return the MvpTfliteMicroAcceleratorWrapper
     * 
     */
    m.def("get_accelerator_wrapper", []() -> void*
    {
        return (void*)&mvp_accelerator;
    });

    /*************************************************************************************************
     * Enable/disabe the simulator backend.
     * This is used internally
     * 
     */
    m.def("set_simulator_backend_enabled", [](bool enabled) -> void
    {
        sli_mvp_set_simulator_backend_enabled(enabled);
    });

    /*************************************************************************************************
     * Enable/disabe only calculating accelerator cycles during simulation
     * This is used internally
     * 
     */
    m.def("set_calculate_accelerator_cycles_only_enabled", [](bool enabled) -> void
    {
       mltk::mvpv1_calculate_accelerator_cycles_only = enabled;
    });
   
    /*************************************************************************************************
     * Enable recording hardware accelerator data/instructions
     * 
     */
    m.def("enable_recorder", []() -> void
    {
       mltk::TfliteMicroAcceleratorRecorder::instance().set_enabled();
    });

    /*************************************************************************************************
     * Return the recorded hardware accelerator data/instructions
     * 
     * This returns: List[Dict[str, List[bytes]]]
     */
    m.def("get_recorded_data", []() -> py::list
    {
        py::list ret_list;

        auto& recorder = mltk::TfliteMicroAcceleratorRecorder::instance();

        for(auto& recorded_layer : recorder)
        {
            py::dict ret_dict;

            for(auto it = recorded_layer.items(); it != recorded_layer.enditems(); ++it)
            {
                const auto dict_item = *it;
                const auto recorded_buffer_list = dict_item->value;
                py::list ret_dict_list;

                for(auto& buffer_list_data : *recorded_buffer_list)
                {
                    std::string buf((const char*)buffer_list_data.data, buffer_list_data.length);
                    ret_dict_list.append(py::bytes(buf));
                }
                
                ret_dict[dict_item->key] = ret_dict_list;
            }

            ret_list.append(ret_dict);
        }

        recorder.clear();

        return ret_list;
    });

}