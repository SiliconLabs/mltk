#pragma once 

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "sl_tflite_micro_opcode_resolver.h"
#include "sl_tflite_micro_model.h"

const auto mltk_model_flatbuffer = sl_tflite_model_array;
const auto mltk_model_flatbuffer_length = sl_tflite_model_len;


const int32_t mltk_model_buffer_count = 1;
uint8_t* mltk_model_buffers[1] = { nullptr };
const int32_t mltk_model_buffer_sizes[1] = { -1 };