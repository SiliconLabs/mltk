#include "mltk_tflite_micro_accelerator.hpp"

namespace mltk
{

extern "C" TfliteMicroAccelerator* mltk_tflite_micro_register_accelerator()
{
    // This is just a placeholder since there is not specific accelerator to register for this CMSIS kernels
    return mltk_tflite_micro_set_accelerator(nullptr);
}

} // namespace mltk