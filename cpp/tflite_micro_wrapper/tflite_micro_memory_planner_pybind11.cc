#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>


#include "tensorflow/lite/micro/memory_planner/greedy_memory_planner.h"


namespace py = pybind11;


class TfliteMicroMemoryPlannerWrapper : public tflite::GreedyMemoryPlanner
{
public:

    TfliteMicroMemoryPlannerWrapper() 
    {
        Init(_scratch_buffer, sizeof(_scratch_buffer));
    }

    bool add_buffer(
        int size, 
        int first_time_used, 
        int last_time_used,
        int offline_offset = -1,
        int subgraph_id = -1,
        int tensor_id = -1
    )
    {
        TfLiteStatus status;

        _buffer_subgraph_ids.push_back(subgraph_id);
        _buffer_tensor_ids.push_back(tensor_id);

        if(offline_offset != -1)
        {
            status = AddBuffer(
                size, 
                first_time_used, 
                last_time_used, 
                offline_offset
            );
        } 
        else 
        {
            status = AddBuffer(
                size, 
                first_time_used, 
                last_time_used
            );
        }

        return status == kTfLiteOk;
    }

    std::vector<std::map<std::string,int>> get_plan() 
    {
        std::vector<std::map<std::string,int>> retval;

        CalculateOffsetsIfNeeded();

        for(int i = 0; i < GetBufferCount(); ++i)
        {
            const auto& req = requirements_[i];
            const int offset = buffer_offsets_[i];

            std::map<std::string,int> buffer;
            buffer["size"] = req.size;
            buffer["start"] = req.first_time_used;
            buffer["end"] = req.last_time_used;
            buffer["offset"] = buffer_offsets_[i];
            buffer["subgraph_id"] = _buffer_subgraph_ids[i];
            buffer["tensor_id"] = _buffer_tensor_ids[i];

            retval.push_back(buffer);
        }

        return retval;
    }

private:
    uint8_t _scratch_buffer[1024*128];
    std::vector<int> _buffer_subgraph_ids;
    std::vector<int> _buffer_tensor_ids;
};



void init_tflite_micro_memory_planner(py::module &m)
{
    py::class_<TfliteMicroMemoryPlannerWrapper>(m, "TfliteMicroMemoryPlannerWrapper")
    .def(py::init<>())
    .def("add_buffer", &TfliteMicroMemoryPlannerWrapper::add_buffer)
    .def("get_plan", &TfliteMicroMemoryPlannerWrapper::get_plan)
    ;
}