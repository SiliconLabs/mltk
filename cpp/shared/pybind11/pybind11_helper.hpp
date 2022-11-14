#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tflite_micro_model/tflite_micro_tensor.hpp"

namespace mltk
{

namespace py = pybind11;



/*************************************************************************************************/
static inline std::string tflite_type_to_format_descriptor(TfLiteType type)
{
    switch(type)
    {
        case kTfLiteInt8: 
            return py::format_descriptor<int8_t>::format();
        case kTfLiteUInt8: 
            return py::format_descriptor<uint8_t>::format();
        case kTfLiteInt16:
            return py::format_descriptor<int16_t>::format();
        case kTfLiteInt32: 
            return py::format_descriptor<int32_t>::format();
        case kTfLiteInt64: 
            return py::format_descriptor<int64_t>::format();
        case kTfLiteFloat32: 
            return py::format_descriptor<float>::format();
        case kTfLiteFloat64: 
            return py::format_descriptor<double>::format();
        default:  
            throw std::invalid_argument("Tensor data type not supported");
    }
}


/*************************************************************************************************/
static inline std::vector<int32_t> runtime_shape_to_vector(const tflite::RuntimeShape& shape)
{
    std::vector<int32_t> vec;
    for(int i = 0; i < shape.DimensionsCount(); ++i)
    {
        vec.push_back(shape.Dims(i));
    }

    return vec;
}


/*************************************************************************************************/
template<typename dtype>
py::array tflite_tensor_to_array(
    const tflite::RuntimeShape& shape,
    dtype* data
)
{
    if(data == nullptr)
    {
        return py::none();
    }

    const auto element_size = sizeof(dtype);
    const int n_dims = shape.DimensionsCount();
    std::vector<ssize_t> dims(n_dims);
    std::vector<ssize_t> strides(n_dims);
    ssize_t stride = 1;
    for(int i = n_dims-1; i >= 0; --i)
    {
        dims[i] = shape.Dims(i);
        strides[i] = stride * element_size;
        stride *= dims[i];
    }

    // We want to return a numpy array object for the 
    // given model tensor WITHOUT copying the data.
    // This way, Python can modify the model tensor.
    // We do this by passing in a dummy py::capsule
    // object which causes the py::array() constructor
    // to not do a copy.
    py::capsule dummy([](){});

    return py::array(py::buffer_info(
            (void*)data,
            element_size,
            py::format_descriptor<dtype>::format(),
            n_dims,
            dims,
            strides
    ), dummy);
}



/*************************************************************************************************/
static inline py::array tflite_tensor_to_array(const TfliteTensorView& tensor)
{
    if(tensor.data.raw == nullptr)
    {
        return py::none();
    }

    const auto shape = tensor.shape();
    const auto element_size = tensor.element_size();
    std::vector<ssize_t> dims(shape.length);
    std::vector<ssize_t> strides(shape.length);
    ssize_t stride = 1;
    for(int i = shape.length-1; i >= 0; --i)
    {
        dims[i] = shape[i];
        strides[i] = stride * element_size;
        stride *= shape[i];
    }

    // We want to return a numpy array object for the 
    // given model tensor WITHOUT copying the data.
    // This way, Python can modify the model tensor.
    // We do this by passing in a dummy py::capsule
    // object which causes the py::array() constructor
    // to not do a copy.
    py::capsule dummy([](){});

    return py::array(py::buffer_info(
            tensor.data.raw,
            element_size,
            tflite_type_to_format_descriptor(tensor.type),
            shape.length,
            dims,
            strides
    ), dummy);
}


/*************************************************************************************************/
template<typename dtype>
py::array create_1d_array(int length, dtype* data)
{
    std::vector<ssize_t> dims{length};
    std::vector<ssize_t> strides{sizeof(dtype)};

    // We want to return a numpy array object for the 
    // given model tensor WITHOUT copying the data.
    // This way, Python can modify the model tensor.
    // We do this by passing in a dummy py::capsule
    // object which causes the py::array() constructor
    // to not do a copy.
    py::capsule dummy([](){});

    return py::array(py::buffer_info(
            (void*)data,
            sizeof(dtype),
            py::format_descriptor<dtype>::format(),
            1,
            dims,
            strides
    ), dummy);
}




} // namespace mltk