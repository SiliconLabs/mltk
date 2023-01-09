/***
 * Voice Activity Detection
 * 
 * This is used to detect voice activity in streaming audio.
 * This implementation is based on: https://github.com/wiseman/py-webrtcvad
 * 
*/

#pragma once 

#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef void* VoiceActivityContext;


/**
 * Return the size in bytes required by the VoiceActivityContext
*/
int voice_activity_detector_context_size();

/**
 * Initialize the given VoiceActivityContext
 * 
 * @param context The pointer to a buffer of at least @ref voice_activity_detector_context_size() bytes
 * 
 * @return 0 if successfully, else init failed
*/
int voice_activity_detector_init(VoiceActivityContext context);

/**
 * Reset the frame processing context
 * 
 * @param context A previously initialized context to reset
*/
void voice_activity_detector_reset(VoiceActivityContext context);


/**
 * Sets the VAD operating mode. A more aggressive (higher mode) VAD is more
 * restrictive in reporting speech. Put in other words the probability of being
 * speech when the VAD returns 1 is increased with increasing mode. As a
 * consequence also the missed detection rate goes up.
 * 
 * @param context A previously initialized context
 * @param model Aggressiveness mode (0, 1, 2, or 3).
 * @return 0 if success, else failure
*/
int voice_activity_detector_set_mode(VoiceActivityContext context, int mode);


/**
 * Calculates a VAD decision for the |audio_frame|. 
 * The given audio frame must be 20ms of 16kHz PCM.
 * 
 * @param context A previously initialized context
 * @param audio_frame 20ms, 16kHz PCM audio
 * @return 1 -> voice detected, 0 -> non-voice, -1 -> error
*/
int voice_activity_detector_process_frame(
    VoiceActivityContext context,
    const int16_t* audio_frame
);


/**
 * Calculates the VAD for the given audio buffer.
 * The audio must be 16kHz, PCM audio.
 * 
 * This will processing `processing_length_ms` of audio.
 * and count the number of 20ms frames that contain VAD (see @ref voice_activity_detector_process_frame()).
 * If the number of VAD frames / processed frames > threshold
 * then voice activity is considered to be in the given audio.
 * 
 * This may be called multiple times, but any remaining audio that is not aligned 
 * to 20ms will be dropped.
 * 
 * Use @ref voice_activity_detector_reset() to reset the processing context.
 * 
 * @param context A previously initialized context
 * @param audio Audio buffer containing 16kHz, PCM audio
 * @param length Number of samples in the given audio buffer
 * @param threshold Ratio of VAD frames to non-VAD frames in previous processing_length_ms 
 *  of audio for data to be considered to contain voice activity.
 *  A larger value means there should be more activity in audio for a detection to triggered
 * @param processing_length_ms The audio length, in milliseconds, to process for activity.
 *  This has an upper limit of 64 * 0.020 = 1280 milliseconds
 * @return  1 -> voice activity detected, 0 -> no voice activity detected, -1 -> less than processing_length_ms of audio processed
*/
int voice_activity_detector_process(
    VoiceActivityContext context,
    const int16_t* audio,
    int length,
    float threshold,
    int processing_length_ms
);



#ifdef __cplusplus
}
#endif