
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#include "sl_status.h"
#include "mltk_sl_mic.h"
#include "soundio/soundio.h"
#include "cpputils/string.hpp"


constexpr const int MAX_READ_LENGTH_SECONDS = 1;


static bool find_microphone_device();
static void microphone_read_callback(struct SoundIoInStream *instream, int frame_count_min, int frame_count_max);


static struct 
{
    struct SoundIo *soundio = nullptr;
    struct SoundIoDevice *selected_device = nullptr;
    struct SoundIoInStream *instream = nullptr;
    uint32_t sample_rate_hz;     
    uint32_t n_channels;
    int16_t* buffer;     
    int16_t* buffer_ptr;
    int16_t* buffer_end;  
    sl_mic_buffer_ready_callback_t ready_callback;
} context;



/*************************************************************************************************/
extern "C" sl_status_t mltk_sl_mic_init(uint32_t sample_rate, uint8_t channels)
{
    int err;
    sl_status_t status = SL_STATUS_FAIL;

    assert(context.soundio == nullptr);

    if(channels != 1)
    {
        printf("Only 1 channel is currently supported\n");
        return SL_STATUS_FAIL;
    }

    context.sample_rate_hz = sample_rate;
    context.n_channels = channels;


    context.soundio = soundio_create();
    if (context.soundio == nullptr)
    {
        goto exit;
    }

    err = soundio_connect(context.soundio);

    if (err != SoundIoErrorNone)
    {
        printf("Failed to start microphone library, err: %d\n", err);
        goto exit;
    }

    if(!find_microphone_device())
    {
        printf("Failed to find compatible microphone device\n");
        goto exit;
    }

    context.instream = soundio_instream_create(context.selected_device);
    if (context.instream == nullptr)
    {
        printf("Failed to create microphone input stream\n");
        goto exit;
    }

    context.instream->software_latency = 0.1f;
    context.instream->format = SoundIoFormatS16LE;
    // We adjust the sample rate in software, see microphone_read_callback()
    context.instream->sample_rate = 0; 
    context.instream->read_callback = microphone_read_callback;

    err = soundio_instream_open(context.instream);
    if (err != SoundIoErrorNone)
    {
        printf("Failed to open microphone input stream, err: %d\n", err);
        goto exit;
    }

    status = SL_STATUS_OK;

exit:
    if(status != SL_STATUS_OK)
    {
        mltk_sl_mic_deinit();
    }

    return status;
}

/*************************************************************************************************/
extern "C" sl_status_t mltk_sl_mic_deinit(void)
{
    if(context.instream != nullptr)
    {
        soundio_instream_destroy(context.instream);
        context.instream = nullptr;
    }
    if(context.selected_device)
    {
        soundio_device_unref(context.selected_device);
        context.selected_device = nullptr;
    }
    if(context.soundio != nullptr)
    {
        soundio_destroy(context.soundio);
        context.soundio = nullptr;
    }
    return SL_STATUS_OK;
}

/*************************************************************************************************/
extern "C" sl_status_t mltk_sl_mic_start_streaming(void *buffer, uint32_t n_frames, sl_mic_buffer_ready_callback_t callback)
{
    if(context.soundio == nullptr)
    {
        return SL_STATUS_NOT_INITIALIZED;
    }

    context.buffer = (int16_t*)buffer;
    context.buffer_ptr = context.buffer;
    // The GSDK sl_mic lib using ping-poinging 
    // and expects the input buffer is double the given n_frames, hence the *2
    context.buffer_end = context.buffer + (n_frames * context.n_channels * 2);
    context.ready_callback = callback;

    int err = soundio_instream_start(context.instream);
    if(err != SoundIoErrorNone)
    {
        printf("Failed to start audio input stream, err: %d\n", err);
        return SL_STATUS_FAIL;
    }

    return SL_STATUS_OK;
}


/*************************************************************************************************/
static bool find_microphone_device()
{
    soundio_flush_events(context.soundio);

    const char* MLTK_MICROPHONE = getenv("MLTK_MICROPHONE");
    int input_count = soundio_input_device_count(context.soundio);

    if(MLTK_MICROPHONE != nullptr)
    {
        printf("Environment variable: MLTK_MICROPHONE=%s\n", MLTK_MICROPHONE);
    }
    printf("Searching for compatible microphones (%d available) ...\n", input_count);
    fflush(stdout);
   
    int compatible_count = 0;
    for (int i = 0; i < input_count; i += 1)
    {
        struct SoundIoDevice *device = soundio_get_input_device(context.soundio, i);
        for (int j = 0; j < device->sample_rate_count; j += 1)
        {
            struct SoundIoSampleRateRange *range = &device->sample_rates[j];
            if(context.sample_rate_hz <= range->max)
            {
                for (int j = 0; j < device->format_count; j += 1)
                {
                    if(device->formats[j] == SoundIoFormatS16LE)
                    {
                        ++compatible_count;
                        if(MLTK_MICROPHONE != nullptr)
                        {
                            if(strcasestr(device->name, MLTK_MICROPHONE) != nullptr)
                            {
                                context.selected_device = device;
                                goto exit;
                            }
                        }
                        else
                        {
                            context.selected_device = device;
                            goto exit;
                        }
                    }
                }
            }
        }

        soundio_device_unref(device);
    }

exit:
    if(context.selected_device != nullptr)
    {
        printf("Using microphone: %s\n", context.selected_device->name);
    }
    else 
    {
        if(compatible_count > 0)
        {
            printf("No compatible microphone found!\nAvailable microphones:\n");
            for (int i = 0; i < input_count; i += 1)
            {
                struct SoundIoDevice *device = soundio_get_input_device(context.soundio, i);
                for (int j = 0; j < device->sample_rate_count; j += 1)
                {
                    struct SoundIoSampleRateRange *range = &device->sample_rates[j];
                    if(context.sample_rate_hz <= range->max)
                    {
                        for (int j = 0; j < device->format_count; j += 1)
                        {
                            if(device->formats[j] == SoundIoFormatS16LE)
                            {
                                printf("- %s\n", device->name);
                                break;
                            }
                        }
                    }
                }

                soundio_device_unref(device);
            }
        }
        else
        {
            printf("No compatible microphone found!\n");
        }
    }

    if(MLTK_MICROPHONE == nullptr)
    {
        printf("(Hint: Manually specify microphone with ENV variable: MLTK_MICROPHONE=<mic-name>)\n\n");
    }
    fflush(stdout);

    return context.selected_device != nullptr;
}

/*************************************************************************************************/
static void microphone_read_callback(struct SoundIoInStream *instream, int frame_count_min, int frame_count_max)
{
    struct SoundIoChannelArea *areas;
    int err;

    int16_t* write_ptr = context.buffer_ptr;
    const int skip_count = instream->sample_rate / context.sample_rate_hz;
    const int write_frames = frame_count_max/skip_count;
    int frames_left = write_frames*skip_count;
    int frames_written = 0;

    for (;;)
    {
        int frame_count = frames_left;

        if ((err = soundio_instream_begin_read(instream, &areas, &frame_count)))
        {
            printf("read begin err, %d\n", err);
            return;
        }

        if (!frame_count)
            break;

        if (!areas)
        {
            // Due to an overflow there is a hole. Fill the ring buffer with
            // silence for the size of the hole.
            memset(write_ptr, 0, frame_count * instream->bytes_per_frame);
            printf("hole in libsoundio buffer detected\n");
        }
        else
        {
            for (int frame = 0; frame < frame_count; frame += skip_count)
            {
                const int16_t *p = (int16_t*)areas[0].ptr;
                *write_ptr++ = *p;
                ++frames_written;
                if(write_ptr >= context.buffer_end)
                {
                    write_ptr = context.buffer;
                }
                areas[0].ptr += (areas[0].step * skip_count);
            }
        }

        if ((err = soundio_instream_end_read(instream)))
        {
            printf("read err, %d\n", err);
            return;
        }

        frames_left -= frame_count;
        if (frames_left <= 0)
        {
            break;
        }
    }

    if(context.ready_callback != nullptr)
    {
        context.ready_callback(context.buffer_ptr, frames_written);
    }
    context.buffer_ptr = write_ptr;
}

