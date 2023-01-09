#include <string.h>
#include <assert.h>

#include "voice_activity_detector.h"
#include "webrtc_vad.h"
#include "webrtc/common_audio/vad/vad_core.h"


#ifndef MIN
#define MIN(x,y)  ((x) < (y) ? (x) : (y))
#endif /* ifndef MIN */


const int FRAME_LENGTH_20MS_16K = ((16000 * 20) / 1000);
const int MAX_PROCESSING_FRAMES = FRAME_LENGTH_20MS_16K * 64;


typedef struct 
{
    VadInstT vad;
    uint64_t frame_results;
    uint8_t n_frames;
} _Context;


MaxAbsValueW16 WebRtcSpl_MaxAbsValueW16;
MaxAbsValueW32 WebRtcSpl_MaxAbsValueW32;
MaxValueW16 WebRtcSpl_MaxValueW16;
MaxValueW32 WebRtcSpl_MaxValueW32;
MinValueW16 WebRtcSpl_MinValueW16;
MinValueW32 WebRtcSpl_MinValueW32;
CrossCorrelation WebRtcSpl_CrossCorrelation;
DownsampleFast WebRtcSpl_DownsampleFast;
ScaleAndAddVectorsWithRound WebRtcSpl_ScaleAndAddVectorsWithRound;




int voice_activity_detector_context_size()
{
    return sizeof(_Context);
}

int voice_activity_detector_init(VoiceActivityContext context)
{
    memset(context, 0, sizeof(_Context));
    WebRtcSpl_MaxAbsValueW16 = WebRtcSpl_MaxAbsValueW16C;
    WebRtcSpl_MaxAbsValueW32 = WebRtcSpl_MaxAbsValueW32C;
    WebRtcSpl_MaxValueW16 = WebRtcSpl_MaxValueW16C;
    WebRtcSpl_MaxValueW32 = WebRtcSpl_MaxValueW32C;
    WebRtcSpl_MinValueW16 = WebRtcSpl_MinValueW16C;
    WebRtcSpl_MinValueW32 = WebRtcSpl_MinValueW32C;
    WebRtcSpl_CrossCorrelation = WebRtcSpl_CrossCorrelationC;
    WebRtcSpl_DownsampleFast = WebRtcSpl_DownsampleFastC;
    WebRtcSpl_ScaleAndAddVectorsWithRound = WebRtcSpl_ScaleAndAddVectorsWithRoundC;

    return WebRtcVad_InitCore(context);
}

void voice_activity_detector_reset(VoiceActivityContext context)
{
    _Context* ctx = (_Context*)context;
    ctx->n_frames = 0;
    ctx->frame_results = 0;
}

int voice_activity_detector_set_mode(VoiceActivityContext context, int mode)
{
    return WebRtcVad_set_mode_core(context, mode);
}


int voice_activity_detector_process_frame(
    VoiceActivityContext context,
    const int16_t* audio_frame
)
{
    int vad;
    vad = WebRtcVad_CalcVad16khz(context, audio_frame, FRAME_LENGTH_20MS_16K);

    if (vad > 0) 
    {
        vad = 1;
    }
    return vad;
}


int voice_activity_detector_process(
    VoiceActivityContext context,
    const int16_t* audio,
    int length,
    float threshold,
    int processing_length_ms
)
{
    _Context* ctx = (_Context*)context;
    int n_frames = length / FRAME_LENGTH_20MS_16K;
    for(int i = 0; i < n_frames; ++i)
    {
        const int is_voice = voice_activity_detector_process_frame(context, audio);
        ctx->frame_results <<= 1;
        ctx->frame_results += is_voice;
        ctx->n_frames = MIN(ctx->n_frames + 1, MAX_PROCESSING_FRAMES);

        audio += FRAME_LENGTH_20MS_16K;
    }

    const int processing_length = (16000 * processing_length_ms) / 1000;
    const int processing_frames = processing_length / FRAME_LENGTH_20MS_16K;
    assert(processing_frames <= MAX_PROCESSING_FRAMES);

    if(ctx->n_frames < processing_frames)
    {
        return -1;
    }
    
    int voice_frames = 0;
    int remaining = processing_frames;
    for(uint64_t results = ctx->frame_results; remaining > 0; --remaining, results >>= 1)
    {
        if(results & 0x01)
        {
            voice_frames += 1;
        }
    }

    const float voice_ratio = (float)voice_frames / (float)processing_frames;
    return voice_ratio >= threshold ? 1 : 0;
}
