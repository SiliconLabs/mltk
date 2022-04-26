#include <cstdint>

#include "jlink_stream/jlink_stream.hpp"
#include "sl_ml_audio_feature_generation_config.h"

extern "C"
{

static bool registered_streams = false;


void register_dump_streams()
{
    if(registered_streams) 
    {
        return;
    }
    registered_streams = true;

    
    if(SL_ML_AUDIO_FEATURE_GENERATION_DUMP_AUDIO)
    {
        jlink_stream::register_stream("audio", jlink_stream::Write);
    }
    
    if(SL_ML_AUDIO_FEATURE_GENERATION_DUMP_RAW_SPECTROGRAM)
    {
        jlink_stream::register_stream("raw_spec", jlink_stream::Write);
    }
    
    if(SL_ML_AUDIO_FEATURE_GENERATION_DUMP_QUANTIZED_SPECTROGRAM)
    {
        jlink_stream::register_stream("quant_spec", jlink_stream::Write);
    }
}


void dump_audio(const int16_t* buffer, int length)
{
    jlink_stream::write_all("audio", buffer, sizeof(int16_t)*length);
}


void dump_raw_spectrogram(const uint16_t* buffer, int length)
{
    jlink_stream::write_all("raw_spec", buffer, sizeof(uint16_t)*length);
}


void dump_quantized_spectrogram(const int8_t* buffer, int length)
{
    jlink_stream::write_all("quant_spec", buffer, length);
}


} // extern "C"