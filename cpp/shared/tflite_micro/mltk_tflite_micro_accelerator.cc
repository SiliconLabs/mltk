#include "mltk_tflite_micro_accelerator.hpp"


namespace mltk 
{


#ifndef TFLITE_MICRO_ACCELERATOR
extern "C" TfliteMicroAccelerator* mltk_tflite_micro_register_accelerator()
{
    // This is just a placeholder if no accelerator is built into the binary / shared library
    // (i.e. each accelerator also defines this API and internally calls mltk_tflite_micro_set_accelerator())
    return mltk_tflite_micro_set_accelerator(nullptr);
}
#endif

#ifndef MLTK_DLL_IMPORT
static TfliteMicroAccelerator* _registered_accelerator = nullptr;
extern "C" TfliteMicroAccelerator* mltk_tflite_micro_set_accelerator(TfliteMicroAccelerator* accelerator)
{
    _registered_accelerator = accelerator;
    return accelerator;
}

extern "C" TfliteMicroAccelerator* mltk_tflite_micro_get_registered_accelerator()
{
    return _registered_accelerator;
}
#endif // MLTK_DLL_IMPORT





} // namespace mltk 