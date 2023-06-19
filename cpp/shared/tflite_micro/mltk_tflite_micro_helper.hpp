#pragma once
#include <cstdio>
#include <cassert>
#include <functional>
#include <complex>


#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/compatibility.h"

#include "float16.hpp"
#include "cpputils/helpers.hpp"
#include "profiling/profiler.hpp"
#include "logging/logger.hpp"



#ifndef MLTK_LOG_LEVEL
#define MLTK_LOG_LEVEL 0
#endif

#if MLTK_LOG_LEVEL <= 0
#define MLTK_DEBUG(msg, ...) ::mltk::get_logger().debug(msg, ## __VA_ARGS__)
#endif
#if MLTK_LOG_LEVEL <= 1
#define MLTK_INFO(msg, ...) ::mltk::get_logger().info(msg, ## __VA_ARGS__)
#endif
#if MLTK_LOG_LEVEL <= 2
#define MLTK_WARN(msg, ...) ::mltk::get_logger().warn(msg, ## __VA_ARGS__)
#endif
#if MLTK_LOG_LEVEL <= 3
#define MLTK_ERROR(msg, ...) ::mltk::get_logger().error(msg, ## __VA_ARGS__)
#endif


#ifndef MLTK_DEBUG
#define MLTK_DEBUG(...)
#endif
#ifndef MLTK_INFO
#define MLTK_INFO(...)
#endif
#ifndef MLTK_WARN
#define MLTK_WARN(...)
#endif
#ifndef MLTK_ERROR
#define MLTK_ERROR(...)
#endif


#define ALLOCATE_PERSISTENT_BUFFER(type, count) ::mltk::allocate_persistent_buffer<type>(context, count)
#define ALLOCATE_SCRATCH_BUFFER(size_bytes, scratch_buffer_index) \
{ \
    const auto status = ::mltk::allocate_scratch_buffer(context, size_bytes, scratch_buffer_index); \
    if(status != kTfLiteOk) return status; \
}
#define GET_SCRATCH_BUFFER(type, scratch_buffer_index) ::mltk::get_scratch_buffer<type>(context, scratch_buffer_index)
#define MLTK_KERNEL_UNSUPPORTED_MSG(fmt, ...) ::mltk::issue_unsupported_kernel_message(fmt, ## __VA_ARGS__);




namespace mltk
{

using Logger = logging::Logger;
using LogLevel = logging::Level;

typedef std::complex<float> cfloat_t;
typedef std::complex<float16_t> cfloat16_t;

struct TfliteMicroAccelerator
{
  const char* name;
  void (*init)();
  void (*deinit)();
  int (*get_profiler_loop_count)();
  void (*start_profiler)(int loop_count);
  void (*stop_profiler)(int loop_count);
  void (*start_op_profiler)(int op_idx, profiling::Profiler* profiler);
  void (*stop_op_profiler)(int op_idx, profiling::Profiler* profiler);
  bool (*set_simulator_memory)(const char* region, void* base_address, uint32_t length);
  bool (*invoke_simulator)(const std::function<bool()>&func);
};


extern bool model_profiler_enabled;
extern bool model_tensor_recorder_enabled;
extern bool model_has_unknown_layers;
extern const char* TFLITE_MICRO_VERSION;


extern "C" DLL_EXPORT void issue_unsupported_kernel_message(const char* fmt, ...);
extern "C" DLL_EXPORT void flush_unsupported_kernel_messages(logging::Level level = logging::Warn);
extern "C" DLL_EXPORT void reset_unsupported_kernel_messages();
extern "C" DLL_EXPORT bool has_unsupported_kernel_messages();
extern "C" DLL_EXPORT void set_unsupported_kernel_messages_enabled(bool enabled);
extern "C" DLL_EXPORT void mltk_tflite_micro_set_accelerator(const TfliteMicroAccelerator* accelerator);
extern "C" DLL_EXPORT const TfliteMicroAccelerator* mltk_tflite_micro_get_registered_accelerator();
extern "C" DLL_EXPORT void mltk_tflite_micro_get_current_layer_opcode_and_index(int* opcode, int* index);
extern "C" void mltk_tflite_micro_register_accelerator();



Logger& get_logger();
bool set_log_level(LogLevel level);
TfLiteStatus allocate_scratch_buffer(TfLiteContext *ctx, unsigned size_bytes, int *scratch_buffer_index);
const void* get_metadata_from_tflite_flatbuffer(const void* tflite_flatbuffer, const char* tag, uint32_t* length = nullptr);
bool get_tflite_flatbuffer_from_end_of_flash(const uint8_t** tflite_flatbuffer, uint32_t* length=nullptr, const uint32_t* flash_end_addr=nullptr);
const char* to_str(tflite::BuiltinOperator op_type);
const char* op_to_str(int op_idx, tflite::BuiltinOperator op_type);



/*************************************************************************************************/
static inline const char* get_current_layer_str()
{
    int opcode;
    int index;
    mltk_tflite_micro_get_current_layer_opcode_and_index(&opcode, &index);
    return op_to_str(index, (tflite::BuiltinOperator)opcode);
}

/*************************************************************************************************/
template<typename T>
T* allocate_persistent_buffer(TfLiteContext *ctx, unsigned count)
{
    return reinterpret_cast<T*>(ctx->AllocatePersistentBuffer(ctx, sizeof(T) * count));
}

/*************************************************************************************************/
template<typename T>
T* get_scratch_buffer(TfLiteContext *ctx, int scratch_buffer_index)
{
    return reinterpret_cast<T*>(ctx->GetScratchBuffer(ctx, scratch_buffer_index));
}

/*************************************************************************************************/
inline tflite::PaddingType runtime_padding_type(TfLitePadding padding)
{
    switch (padding)
    {
    case TfLitePadding::kTfLitePaddingSame:
        return tflite::PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
        return tflite::PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
        return tflite::PaddingType::kNone;
    }
}

/*************************************************************************************************/
inline int div_round_up(const int numerator, const int denominator)
{
    return (numerator + denominator - 1) / denominator;
}

/*************************************************************************************************/
inline int div_ceil(const int numerator, const int denominator, int lower_limit = std::numeric_limits<int>::min())
{
    assert(denominator > 0);
    const std::div_t div = std::div(numerator, denominator);

    const int ceil = div.quot + (div.rem ? 1 : 0); // ceil

    return std::max(ceil, lower_limit);
}

/*************************************************************************************************/
inline int div_floor(const int numerator, const int denominator, int upper_limit = std::numeric_limits<int>::max())
{
    assert(denominator > 0);
    const std::div_t div = std::div(numerator, denominator);

    const int floor = div.quot; // floor

    return std::min(floor, upper_limit);
}


bool verify_model_flatbuffer(const void* flatbuffer, int flatbuffer_length);


} // namespace mltk
