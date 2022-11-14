#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

#include <cstdarg>
#include <cstdint>
#include <new>

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_string.h"
#endif

#include "logging/logging.hpp"
#include "mltk_tflite_micro_helper.hpp"


namespace mltk 
{
    bool model_error_reporter_enabled = true;
}




void Log(const char* format, va_list args) {
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
  if(mltk::model_error_reporter_enabled)
  {
    auto& logger = mltk::get_logger();
    const auto orig_flags = logger.flags();
    logger.flags().clear(logging::Newline);
    logger.vwrite(logging::Error, format, args);
    logger.write(logging::Error, "\n");
    logger.flags(orig_flags);
  }
#endif
}

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
void MicroPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  Log(format, args);
  va_end(args);
}
#endif

