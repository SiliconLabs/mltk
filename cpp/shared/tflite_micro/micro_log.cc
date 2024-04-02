#include "tensorflow/lite/micro/micro_log.h"

#include <cstdarg>
#include <cstdint>
#include "mltk_tflite_micro_helper.hpp"

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
#include "tensorflow/lite/micro/debug_log.h"
#endif


void MicroPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  VMicroPrintf(format, args);
  va_end(args);
}

int MicroSnprintf(char* buffer, size_t buf_size, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int result = MicroVsnprintf(buffer, buf_size, format, args);
  va_end(args);
  return result;
}

int MicroVsnprintf(char* buffer, size_t buf_size, const char* format,
                   va_list vlist) {
  return vsnprintf(buffer, buf_size, format, vlist);
}


void VMicroPrintf(const char* format, va_list args) {
  auto& logger = mltk::get_logger();
  const auto orig_flags = logger.flags();
  logger.flags().clear(logging::Newline);
  logger.vwrite(logging::Warn, format, args);
  logger.write(logging::Warn, "\n");
  logger.flags(orig_flags);
}