#include <cassert>

#include "mltk_tflite_micro_logger.hpp"



namespace mltk 
{

static Logger *mltk_logger =  nullptr;



Logger& get_logger()
{
    if(mltk_logger == nullptr)
    {
        mltk_logger = logging::get("MLTK");
        if(mltk_logger == nullptr)
        {
            mltk_logger = logging::create("MLTK", LogLevel::Info);
            assert(mltk_logger != nullptr);
        }
    }

    return *mltk_logger;
}


bool set_log_level(LogLevel level)
{
    return get_logger().level(level);
}

} // namespace mltk 