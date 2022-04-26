
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "mltk_tflite_micro_helper.hpp"


namespace mltk
{

class TfliteMicroAcceleratorWrapper
{
public:
    const TfliteMicroAccelerator* accelerator = nullptr;
    
    // Register the accelerator and return its op resolver
    virtual tflite::MicroOpResolver* load() = 0;
};


} // namespace mltk