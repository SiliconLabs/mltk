
#include "all_ops_resolver.h"
#include "mltk_tflite_micro_helper.hpp"
#include "tflite_micro_accelerator_wrapper.hpp"


namespace mltk
{

static tflite::AllOpsResolver mvp_ops_resolver;


class MvpTfliteMicroAcceleratorWrapper : public TfliteMicroAcceleratorWrapper
{
public:
    tflite::MicroOpResolver* load() override
    {
        get_logger().debug("Loading MVP accelerator");
        this->accelerator = mltk_tflite_micro_register_accelerator();
        return &mvp_ops_resolver;
    }
};

} // namespace mltk