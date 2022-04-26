#include <cstdint>
#include <string>


#include "sl_ml_audio_feature_generation_config.h"


#define FEATURE_BUFFER_SLICE_COUNT  (1 + ((SL_ML_FRONTEND_SAMPLE_LENGTH_MS - SL_ML_FRONTEND_WINDOW_SIZE_MS) / SL_ML_FRONTEND_WINDOW_STEP_MS)) 

std::string dump_audio_dir;
std::string dump_raw_spectrograms_dir;
std::string dump_spectrograms_dir;



extern "C"
{



void register_dump_streams()
{
}


void dump_audio(const int16_t* buffer, int length)
{
    static int audio_counter = 0;
    char path[1024];
    const char* audio_dir = dump_audio_dir.c_str();
    if(audio_dir == nullptr)
    {
        return;
    }

    snprintf(path, sizeof(path), "%s/%d.int16.bin", audio_dir, audio_counter++);

    auto fp = fopen(path, "wb");
    if(fp != nullptr)
    {
        fwrite(buffer, sizeof(int16_t), length, fp);
        fclose(fp);
    }
}


void dump_raw_spectrogram(const uint16_t* buffer, int length)
{
    static int spectrogram_counter = 0;
    const char* spectrogram_dir = dump_raw_spectrograms_dir.c_str();
    if(spectrogram_dir == nullptr)
    {
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%d.uint16.npy.txt", spectrogram_dir, spectrogram_counter++);
    auto spectrogram_fp = fopen(path, "wb");
    if(spectrogram_fp != nullptr)
    {
        const uint16_t* ptr = buffer;
        for(int r = 0; r < FEATURE_BUFFER_SLICE_COUNT; ++r)
        {
            for(int c = 0; c < SL_ML_FRONTEND_FILTERBANK_N_CHANNELS; ++c)
            {
                char tmp[64];
                int l = sprintf(tmp, "%d,", *ptr++);
                if(c == SL_ML_FRONTEND_FILTERBANK_N_CHANNELS-1)
                {
                    tmp[l-1] = '\n';
                }
                fwrite(tmp, 1, l, spectrogram_fp);
            }
        }
        fclose(spectrogram_fp);
    }
}


void dump_quantized_spectrogram(const int8_t* buffer, int length)
{
    static int spectrogram_counter = 0;
    const char* spectrogram_dir = dump_spectrograms_dir.c_str();
    if(spectrogram_dir == nullptr)
    {
        return;
    }
    char path[1024];
    snprintf(path, sizeof(path), "%s/%d.int8.npy.txt", spectrogram_dir, spectrogram_counter++);
    auto spectrogram_fp = fopen(path, "wb");
    if(spectrogram_fp != nullptr)
    {
        const int8_t* ptr = buffer;
        for(int r = 0; r < FEATURE_BUFFER_SLICE_COUNT; ++r)
        {
            for(int c = 0; c < SL_ML_FRONTEND_FILTERBANK_N_CHANNELS; ++c)
            {
                char tmp[64];
                int l = sprintf(tmp, "%d,", *ptr++);
                if(c == SL_ML_FRONTEND_FILTERBANK_N_CHANNELS-1)
                {
                    tmp[l-1] = '\n';
                }
                fwrite(tmp, 1, l, spectrogram_fp);
            }
        }
        fclose(spectrogram_fp);
    }
}


} // extern "C"