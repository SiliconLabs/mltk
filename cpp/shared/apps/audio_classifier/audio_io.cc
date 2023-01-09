#include <cassert>
#include <algorithm>
#include <cstdio>

#include "em_device.h"
#include "em_chip.h"
#include "em_cmu.h"
#include "em_emu.h"
#include "em_vdac.h"
#include "em_timer.h"
#include "em_ldma.h"
#include "em_core.h"
#include "dmadrv.h"

#include "audio_classifier_config.h"
#include "sl_ml_audio_feature_generation.h"
#include "sl_ml_audio_feature_generation_config.h"

#include "uart_stream.hpp"
#include "audio_io.h"
#include "opus.h"
#include "voice_activity_detector.h"

#define MINIMP3_ONLY_MP3
#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"



// --------------------------------------------------------
// Command codes to be sync'd with the python script
// --------------------------------------------------------

#define CMD_START_MICROPHONE                          0
#define CMD_STOP_MICROPHONE                           1
#define CMD_START_MICROPHONE_AT_END_OF_SPEAKER_AUDIO  2
#define CMD_START_PLAYBACK                            3


// --------------------------------------------------------
// Speaker settings
// --------------------------------------------------------

#define SPEAKER_VDAC                                  VDAC0
#define SPEAKER_VDAC_CLOCK                            cmuClock_VDAC0
#define SPEAKER_VDAC_CHANNEL_NUM                      0
#define SPEAKER_VDAC_CHANNEL_REG                      SPEAKER_VDAC->CH0F
#define SPEAKER_TIMER                                 TIMER0
#define SPEAKER_TIMER_CLOCK                           cmuClock_TIMER0
#define SPEAKER_TIMER_LDMA_SIGNAL                     ldmaPeripheralSignal_TIMER0_UFOF

#define SPEAKER_FRAMES_PER_DMA                        ALIGN_DOWN(LDMA_MAX_XFER_LENGTH, MPEG2_FRAME_LENGTH)
#define SPEAKER_DMA_COUNT                             8
#define SPEAKER_MP3_BUFFER_LENGTH                     4*1024


// --------------------------------------------------------
// Microphone Settings
// --------------------------------------------------------

#define MICROPHONE_SAMPLE_RATE                        16000
#define MICROPHONE_GAIN                               7

#define MICROPHONE_OPUS_LENGTH_MS                     20 // 20ms per AVS spec
#define MICROPHONE_OPUS_FRAME_SIZE                    ((MICROPHONE_SAMPLE_RATE * MICROPHONE_OPUS_LENGTH_MS) / 1000)

#define MICROPHONE_CHUNK_LENGTH_MS                    (MICROPHONE_OPUS_LENGTH_MS * 10)
#define MICROPHONE_CHUNK_LENGTH                       ((MICROPHONE_CHUNK_LENGTH_MS * MICROPHONE_SAMPLE_RATE) / 1000)
#define MICROPHONE_CHUNK_PCM_BYTE_LENGTH              (MICROPHONE_CHUNK_LENGTH*sizeof(int16_t))
#define MICROPHONE_OPUS_BYTE_LENGTH                   80 // 80 bytes per AVS spec
#define MICROPHONE_CHUNK_OPUS_BYTE_LENGTH             (MICROPHONE_OPUS_BYTE_LENGTH * (MICROPHONE_CHUNK_LENGTH_MS / MICROPHONE_OPUS_LENGTH_MS))



// --------------------------------------------------------
// Other helper macros
// --------------------------------------------------------

// Maximum LDMA XFER count (2048)
#define LDMA_MAX_XFER_LENGTH                          ((_LDMA_CH_CTRL_XFERCNT_MASK >> _LDMA_CH_CTRL_XFERCNT_SHIFT)+1)
// MPEG 2 Layer 3 frame length in mono samples
#define MPEG2_FRAME_LENGTH                            576
// Convert from int16 to uint16
// then drop the lower 4-bits as the VDAC only has 12-bit resolution
#define INT16_TO_VDAC(int16)                          ((uint16_t)((((int32_t)(int16)) + INT16_MAX) >> 4))

#define ALIGN_UP(x, n) ((((x) + ((n)-1)) / (n)) * n)
#define ALIGN_DOWN(x, n) (((x) / (n)) * (n))


static bool microphone_init();
static void microphone_process_outgoing_data();

static bool speaker_init();
static void speaker_set_sample_rate(int sample_rate);
static void speaker_start_playback();
static void speaker_stop_playback();
static void speaker_process_incoming_data();


static struct
{
  uint16_t vdac_buffer[SPEAKER_DMA_COUNT][SPEAKER_FRAMES_PER_DMA];
  uint16_t* vdac_buffer_end;
  uint16_t* vdac_buffer_ptr;
  uint8_t mp3_buffer[SPEAKER_MP3_BUFFER_LENGTH];
  int mp3_buffer_used;
  volatile int n_buffered_pcm_samples;
  uint32_t n_processed_pcm_samples;
  int32_t n_mp3_bytes_remaining;
  LDMA_Descriptor_t dma_descriptors[SPEAKER_DMA_COUNT];
  mp3dec_t mp3_decoder;
  int sample_rate;
  TIMER_Prescale_TypeDef prescaler;
  uint8_t uart_rx_buffer[4*1024];
} speaker;


static struct
{
  const int16_t* head;
  const int16_t* tail;
  const int16_t* start;
  const int16_t* end;
  volatile uint32_t n_buffered_pcm_samples;
  uint32_t n_processed_pcm_samples;
  uint8_t opus_encoder_buffer[24544]; // this comes from opus_encoder_get_size(1))
  uint8_t opus_buffer[MICROPHONE_CHUNK_OPUS_BYTE_LENGTH];
  uint8_t vad_context[752]; // this comes from voice_activity_detector_context_size()
  bool vad_detected;
  bool start_at_end_of_speaker_audio;
} microphone;



/*************************************************************************************************/
bool audio_io_init()
{
  if(!uart_stream::initialize(
    speaker.uart_rx_buffer,
    sizeof(speaker.uart_rx_buffer),
    SL_TFLITE_MODEL_BAUD_RATE
  ))
  {
    printf("Failed to init UART stream\n");
    return false;
  }

  if(!speaker_init())
  {
    printf("Failed to init speaker\n");
    return false;
  }


  if(!microphone_init())
  {
    printf("Failed to init microphone\n");
    return false;
  }

  return true;
}

/*************************************************************************************************/
void audio_io_process()
{
    if(uart_stream::synchronize())
    {
      uint8_t cmd;
      uint8_t payload[uart_stream::COMMAND_PAYLOAD_LENGTH];

      // Check if any commands have been sent from the Python script
      if(uart_stream::read_cmd(&cmd, payload))
      {
        switch(cmd)
        {
        case CMD_START_MICROPHONE:
          audio_io_set_microphone_streaming_enabled(true);
          break;
        case CMD_STOP_MICROPHONE:
          audio_io_set_microphone_streaming_enabled(false);
          break;
        case CMD_START_MICROPHONE_AT_END_OF_SPEAKER_AUDIO:
          microphone.start_at_end_of_speaker_audio = true;
          break;
        case CMD_START_PLAYBACK:
          speaker.n_mp3_bytes_remaining += *(int32_t*)payload;
          break;
        }
      }

      microphone_process_outgoing_data();
      speaker_process_incoming_data();
    }
    else
    {
      microphone.start_at_end_of_speaker_audio = false;
      speaker.n_mp3_bytes_remaining = 0;
      speaker.mp3_buffer_used = 0;
      speaker.n_processed_pcm_samples = 0;
    }
}

/*************************************************************************************************/
void audio_io_set_microphone_streaming_enabled(bool enabled)
{
  if(!enabled)
  {
    uart_stream::write_cmd(CMD_STOP_MICROPHONE);
    microphone.head = nullptr;
    microphone.n_buffered_pcm_samples = 0;
  }
  else
  {
    const int16_t *tail;

    CORE_ATOMIC_SECTION(
      tail = microphone.tail;
      microphone.n_buffered_pcm_samples = 0;
    )

    microphone.n_processed_pcm_samples = 0;
    microphone.vad_detected = false;
    microphone.start_at_end_of_speaker_audio = false;
    voice_activity_detector_reset(microphone.vad_context);

    // Ensure that the microphone.head pointer is aligned to an MICROPHONE_OPUS_FRAME_SIZE
    // this way we can cleanly wrap to the beginning of the buffer
    uintptr_t offset = ALIGN_DOWN((uintptr_t)(tail - microphone.start), MICROPHONE_OPUS_FRAME_SIZE);
    microphone.head = microphone.start + offset;
  }
}

/*************************************************************************************************/
static bool microphone_init()
{
  const volatile int opus_encoder_size = opus_encoder_get_size(1);
  assert(sizeof(microphone.opus_encoder_buffer) == opus_encoder_size); (void)opus_encoder_size;
  assert(SL_TFLITE_MODEL_FE_SAMPLE_RATE_HZ == MICROPHONE_SAMPLE_RATE);
  assert(SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_BUFFER_SIZE % MICROPHONE_OPUS_FRAME_SIZE == 0);


  memset(&microphone, 0, sizeof(microphone));

  // This is periodically called by the microphone driver
  // for each buffered chunk of audio
  auto microphone_data_callback = [](const int16_t *buffer, uint32_t n_frames)
  {
    microphone.tail += n_frames;
    if(microphone.tail >= microphone.end)
    {
      microphone.tail = microphone.start;
    }
    microphone.n_buffered_pcm_samples = std::min(
      (int)(microphone.n_buffered_pcm_samples + n_frames),
      (int)SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_BUFFER_SIZE
    );
  };

  sl_ml_audio_feature_generation_set_mic_callback(microphone_data_callback);
  microphone.start = sl_ml_audio_feature_generation_audio_buffer;
  microphone.end = microphone.start + SL_ML_AUDIO_FEATURE_GENERATION_AUDIO_BUFFER_SIZE;
  microphone.tail = microphone.start;
  microphone.head = nullptr;

  auto opcoder_encoder = (OpusEncoder*)microphone.opus_encoder_buffer;
  // NOTE: Using OPUS_APPLICATION_RESTRICTED_LOWDELAY is critical for running on Bobcat
  //       AVS does not seem to mind the slightly degraded audio quality
  int error = opus_encoder_init(opcoder_encoder, MICROPHONE_SAMPLE_RATE, 1, OPUS_APPLICATION_RESTRICTED_LOWDELAY);
  assert(error == 0);

  // These come from:
  // https://developer.amazon.com/en-US/docs/alexa/alexa-voice-service/speechrecognizer.html#recognize
  opus_encoder_ctl(opcoder_encoder, OPUS_SET_BITRATE(32000));
  opus_encoder_ctl(opcoder_encoder, OPUS_SET_COMPLEXITY(4));
  opus_encoder_ctl(opcoder_encoder, OPUS_SET_SIGNAL(OPUS_SIGNAL_VOICE));
  opus_encoder_ctl(opcoder_encoder, OPUS_SET_EXPERT_FRAME_DURATION(OPUS_FRAMESIZE_20_MS));
  opus_encoder_ctl(opcoder_encoder, OPUS_SET_VBR(0));

  const volatile int vad_context_size = voice_activity_detector_context_size();
  assert(sizeof(microphone.vad_context) == vad_context_size); (void)vad_context_size;
  voice_activity_detector_init(microphone.vad_context);

  return true;
}

/*************************************************************************************************
 * This is periodically called in the application loop.
 *
 * It does the following:
 * 1. Read buffered 16-bit, PCM microphone audio
 * 2. Compresses the audio using the Opus Encoder
 * 3. Write the compressed audio to the UART stream
 *
 * This also uses Voice Activity Detection (VAD) to determine when the user finishes speaking a command
 */
static void microphone_process_outgoing_data()
{
  // Check if we should start the microphone at the end of the current speaker audio stream
  if(microphone.start_at_end_of_speaker_audio && speaker.n_buffered_pcm_samples <= SPEAKER_FRAMES_PER_DMA)
  {
    audio_io_set_microphone_streaming_enabled(true);
  }


  // Check if the microphone is enabled and we have enough buffer microphone audio
  const uint32_t n_buffered_pcm_samples = microphone.n_buffered_pcm_samples;
  if(microphone.head == nullptr || n_buffered_pcm_samples < MICROPHONE_CHUNK_LENGTH)
  {
    // Return if the microphone is disable or there isn't enough buffered audio
    return;
  }

  // Determine the amount of audio we can write to the UART stream
  const int32_t bytes_available = uart_stream::get_tx_bytes_available();

  // Ensure we can write at least one "chunk" of compressed audio
  if(bytes_available < MICROPHONE_CHUNK_OPUS_BYTE_LENGTH)
  {
    return;
  }

  // Determine the amount of contiguous microphone audio we can read, compress, and write to UART stream
  const int16_t* microphone_src = microphone.head;
  const uintptr_t length_to_end = microphone.end - microphone_src;
  const int32_t pcm_chunk_length = std::min((int32_t)length_to_end, (int32_t)MICROPHONE_CHUNK_LENGTH);
  assert(pcm_chunk_length != 0 && pcm_chunk_length % MICROPHONE_OPUS_FRAME_SIZE == 0);

  // Update the microphone read buffer pointer
  microphone.head += pcm_chunk_length;
  if(microphone.head >= microphone.end)
  {
    microphone.head = microphone.start;
  }
  // Also update the bookkeeping counters
  microphone.n_processed_pcm_samples += pcm_chunk_length;
  CORE_ATOMIC_SECTION(
    microphone.n_buffered_pcm_samples -= pcm_chunk_length;
  );

  // Increase the volume of the microphone audio if necessary
#ifdef MICROPHONE_GAIN
  int16_t* p = (int16_t*)microphone_src;
  for(int i = pcm_chunk_length; i > 0; --i)
  {
    const int32_t v = *p;
    *p++ = std::max(std::min(v*MICROPHONE_GAIN, (int32_t)32768), (int32_t)-32768);
  }
#endif

  // Compress the microphone audio using the Opus encoder
  auto opcoder_encoder = (OpusEncoder*)microphone.opus_encoder_buffer;
  uint8_t* dst = microphone.opus_buffer;
  const int16_t* src = microphone_src;
  int opus_chunk_length = 0;
  for(int32_t remaining = pcm_chunk_length; remaining > 0; remaining -= MICROPHONE_OPUS_FRAME_SIZE)
  {
    int enc_len = opus_encode(opcoder_encoder, src, MICROPHONE_OPUS_FRAME_SIZE, dst, MICROPHONE_OPUS_BYTE_LENGTH);
    assert(enc_len == MICROPHONE_OPUS_BYTE_LENGTH);
    opus_chunk_length += MICROPHONE_OPUS_BYTE_LENGTH;
    src += MICROPHONE_OPUS_FRAME_SIZE;
    dst += MICROPHONE_OPUS_BYTE_LENGTH;
  }

  // Write the compressed audio to the UART stream
  uart_stream::write(microphone.opus_buffer, opus_chunk_length, false);

  // We need to determine when the user stops speaking the command.
  // We do this by first detecting when the user is activity speaking the command
  // AND THEN when there is no voice activity for a short period of time.
  // Once these conditions are met (or timed-out), then we stop the microphone step
  // to indicate to the Python script that the command has ended.

  // If not voice activity has been detected since the microphone was started
  if(!microphone.vad_detected)
  {
    // Check if voice activity was detected with 80% for previous 350ms
    if(voice_activity_detector_process(microphone.vad_context, microphone_src, pcm_chunk_length, 0.8, 350) == 1)
    {
      // At this point, the user is actively speaking the command
      microphone.vad_detected = 1;
      voice_activity_detector_reset(microphone.vad_context);
    }
    // Otherwise, if there has only been silence for 5s, then stop the microphone
    else if(microphone.n_processed_pcm_samples >= MICROPHONE_SAMPLE_RATE * 5)
    {
      audio_io_set_microphone_streaming_enabled(false);
    }
  }
  else
  {
    // At this point, the user was/is actively speaking the command
    // Detect when there is silence for ~750ms
    if(voice_activity_detector_process(microphone.vad_context, microphone_src, pcm_chunk_length, 0.3, 750) == 0)
    {
      // Silence has been detected for ~750ms, we consider this the end of the command
      // So stop the microphone and notify the Python script
      audio_io_set_microphone_streaming_enabled(false);
    }
    // Otherwise, if there has been continuous non-silence for 15s then just stop the microphone
    else if(microphone.n_processed_pcm_samples >= MICROPHONE_SAMPLE_RATE * 15)
    {
      audio_io_set_microphone_streaming_enabled(false);
    }
  }

}


/*************************************************************************************************/
static bool speaker_init()
{

  memset(&speaker, 0, sizeof(speaker));

  // -------------------------------------------------------------------------
  // MP3 Decoder init
  //
  {
    speaker.sample_rate = -1;
    speaker.vdac_buffer_ptr = speaker.vdac_buffer[0];
    speaker.vdac_buffer_end = speaker.vdac_buffer_ptr + SPEAKER_FRAMES_PER_DMA * SPEAKER_DMA_COUNT;
    mp3dec_init(&speaker.mp3_decoder);
  }

  {
    // -------------------------------------------------------------------------
    // VDAC Init
    //
    // Use default settings
    VDAC_Init_TypeDef        init        = VDAC_INIT_DEFAULT;
    VDAC_InitChannel_TypeDef initChannel = VDAC_INITCHANNEL_DEFAULT;

    // The EM01GRPACLK is chosen as VDAC clock source since the VDAC will be
    // operating in EM1
    CMU_ClockSelectSet(SPEAKER_VDAC_CLOCK, cmuSelect_EM01GRPACLK);
    // Enable the VDAC clocks
    CMU_ClockEnable(SPEAKER_VDAC_CLOCK, true);
    // Calculate the VDAC clock prescaler value resulting in a 1 MHz VDAC clock.
    // (Set the VDAC to max frequency of 1 MHz)
    init.prescaler = VDAC_PrescaleCalc(SPEAKER_VDAC, (uint32_t)1000000);

    // Set reference to internal 2.5V
    init.reference = vdacRef2V5;
    // Since the minimum load requirement for high capacitance mode is 25 nF, turn
    // this mode off
    initChannel.highCapLoadEnable = false;
    initChannel.powerMode = vdacPowerModeHighPower;
    // Initialize the VDAC and VDAC channel
    VDAC_Init(SPEAKER_VDAC, &init);
    VDAC_InitChannel(SPEAKER_VDAC, &initChannel, SPEAKER_VDAC_CHANNEL_NUM);
    // Enable the VDAC
    VDAC_Enable(SPEAKER_VDAC, SPEAKER_VDAC_CHANNEL_NUM, true);
    SPEAKER_VDAC_CHANNEL_REG = INT16_TO_VDAC(0);
  }

  // -------------------------------------------------------------------------
  // TIMER Init
  //
  {
    TIMER_Init_TypeDef init = TIMER_INIT_DEFAULT;
    init.dmaClrAct = true;
    init.enable = false;

    // Enable clock for TIMER module
    CMU_ClockEnable(SPEAKER_TIMER_CLOCK, true);

    TIMER_Init(SPEAKER_TIMER, &init);
    speaker.prescaler = init.prescale;
  }

  // -------------------------------------------------------------------------
  // LDMA Init
  //
  {
    static const LDMA_Descriptor_t desc_template =
    {
      .xfer =
      {
        .structType   = ldmaCtrlStructTypeXfer,
        .structReq    = 0,
        .xferCnt      = SPEAKER_FRAMES_PER_DMA-1,
        .byteSwap     = 0,
        .blockSize    = ldmaCtrlBlockSizeUnit1,
        .doneIfs      = 1,
        .reqMode      = ldmaCtrlReqModeBlock,
        .decLoopCnt   = 0,
        .ignoreSrec   = 0,
        .srcInc       = ldmaCtrlSrcIncOne,
        .size         = ldmaCtrlSizeHalf,
        .dstInc       = ldmaCtrlDstIncNone,
        .srcAddrMode  = ldmaCtrlSrcAddrModeAbs,
        .dstAddrMode  = ldmaCtrlDstAddrModeAbs,
        .srcAddr      = 0,
        .dstAddr      = (uint32_t)&SPEAKER_VDAC_CHANNEL_REG,
        .linkMode     = ldmaLinkModeRel,
        .link         = 1,
        .linkAddr     = 4
      }
    };

    LDMA_Descriptor_t* desc_ptr = speaker.dma_descriptors;

    // For each sample buffer
    for(uint32_t sample_id = 0; sample_id < SPEAKER_DMA_COUNT; ++sample_id)
    {
        *desc_ptr = desc_template;
        desc_ptr->xfer.srcAddr = (uint32_t)&speaker.vdac_buffer[sample_id];
        // Update the counters and pointers
        ++desc_ptr;
    }

    // The last DMA descriptor points to the first descriptor
    // to create a ring between the sample buffers:
    // buffer1 -> buffer2 -> buffer3 -> buffer1 -> ...
    --desc_ptr;
    desc_ptr->xfer.linkAddr = -((SPEAKER_DMA_COUNT-1)*4);


    // Transfer configuration and trigger selection
    // Trigger on TIMER0 overflow and set loop count to size of the sine table
    // minus one
    LDMA_TransferCfg_t transferConfig =
      LDMA_TRANSFER_CFG_PERIPHERAL(SPEAKER_TIMER_LDMA_SIGNAL);

    DMADRV_Init();
    unsigned int ch;
    if(DMADRV_AllocateChannel(&ch, nullptr) != ECODE_EMDRV_DMADRV_OK)
    {
      return false;
    }

    // This is called by the DMADRV each time SPEAKER_FRAMES_PER_DMA samples are written to the VDAC
    auto vdac_dma_callback = [](unsigned int channel, unsigned int sequenceNo, void *userParam)
    {
      CORE_ATOMIC_SECTION(
        speaker.n_buffered_pcm_samples -= SPEAKER_FRAMES_PER_DMA;
        assert(speaker.n_buffered_pcm_samples >= 0);
        // NOTE: We need to stop the TIMER when then less than OR EQUAL to 1 DMA buffer,
        //       as, at this point, another buffer is already playing.
        //       The last buffer should be just silence padding anyways.
        if(speaker.n_buffered_pcm_samples <= SPEAKER_FRAMES_PER_DMA)
        {
          speaker_stop_playback();
        }
      );
      return true;
    };

    DMADRV_LdmaStartTransfer(ch, &transferConfig, speaker.dma_descriptors, vdac_dma_callback, nullptr);
  }

  return true;
}


/*************************************************************************************************/
static void speaker_set_sample_rate(int sample_rate)
{
  if(sample_rate > 0 && speaker.sample_rate != sample_rate)
  {
    uint32_t timerFreq, topValue;

    // Set top (reload) value for the timer
    // Note: the timer runs off of the EM01GRPACLK clock
    timerFreq = CMU_ClockFreqGet(SPEAKER_TIMER_CLOCK) / (speaker.prescaler + 1);
    topValue = (timerFreq / sample_rate);

    // Set top value to overflow at the desired TIMER0_FREQ frequency
    TIMER_TopSet(SPEAKER_TIMER, topValue);

    speaker.sample_rate = sample_rate;
  }
}

/*************************************************************************************************/
static void speaker_start_playback()
{
  // Enable speaker playback by starting the TIMER
  // IFF the timer is not running AND there are enough audio frames buffered
  if(!(SPEAKER_TIMER->STATUS & TIMER_STATUS_RUNNING))
  {
    TIMER_Enable(SPEAKER_TIMER, true);
  }
}

/*************************************************************************************************/
static void speaker_stop_playback()
{
  TIMER_Enable(SPEAKER_TIMER, false);
}

/*************************************************************************************************
 * This is periodically called in the application loop.
 *
 * It does the following:
 * 1. Reads MP3 audio from the UART stream
 * 2. Decompresses the audio
 * 3. Write the decompressed, PCM audio to the VDAC buffer
 */
static void speaker_process_incoming_data()
{
  // Read MP3 audio from the UART stream into the local MP3 buffer
  uint8_t* mp3_ptr = &speaker.mp3_buffer[speaker.mp3_buffer_used];
  int32_t mp3_length = uart_stream::read(mp3_ptr, SPEAKER_MP3_BUFFER_LENGTH - speaker.mp3_buffer_used);
  if(mp3_length < 0)
  {
    return;
  }
  speaker.mp3_buffer_used += mp3_length;


  // Process the MP3 buffer, one MPEG2_FRAME_LENGTH at a time
  // Until we run out of MP3 buffer OR the VDAC buffer becomes too full
  mp3_ptr = speaker.mp3_buffer;
  int mp3_buffer_remaining = speaker.mp3_buffer_used;
  while(mp3_buffer_remaining > 0 && speaker.n_buffered_pcm_samples < SPEAKER_FRAMES_PER_DMA*(SPEAKER_DMA_COUNT-1))
  {
    // Decode the next MPEG2_FRAME_LENGTH
    mp3dec_frame_info_t mp3_info;
    int16_t pcm_samples[MINIMP3_MAX_SAMPLES_PER_FRAME];
    int n_samples = mp3dec_decode_frame(&speaker.mp3_decoder, mp3_ptr, mp3_buffer_remaining, pcm_samples, &mp3_info);
    if(mp3_info.frame_bytes == 0)
    {
      break;
    }

    // Update the sample rate (if necessary)
    speaker_set_sample_rate(mp3_info.hz);

    // Update the bookkeeping counters
    mp3_buffer_remaining -= mp3_info.frame_bytes;
    mp3_ptr += mp3_info.frame_bytes;
    speaker.n_mp3_bytes_remaining -= mp3_info.frame_bytes;
    assert(speaker.n_mp3_bytes_remaining >= 0);

    // If no samples were decoded,
    // then continue to the beginning of the loop
    if(n_samples == 0)
    {
      continue;
    }

    assert(n_samples <= MPEG2_FRAME_LENGTH);
    uint16_t* vdac_ptr = speaker.vdac_buffer_ptr;
    int n_frames_processed = 0;

    // Lambda function to write the int16 PCM value to the VDAC buffer
    auto write_vdac_buffer = [](uint16_t* ptr, int16_t value)
    {
      *ptr++ = INT16_TO_VDAC(value);
      if(ptr >= speaker.vdac_buffer_end)
      {
        ptr = speaker.vdac_buffer[0];
      }
      return ptr;
    };


    // Next write the decoded audio to the VDAC buffer
    const int16_t *src = pcm_samples;
    n_frames_processed += n_samples;
    speaker.n_processed_pcm_samples += n_samples;
    for(int i = n_samples; i > 0; --i)
    {
      vdac_ptr = write_vdac_buffer(vdac_ptr, *src);
      src += mp3_info.channels;
    }

    // If this is the end of the MP3 audio
    // then ensure the VDAC buffer pointer is aligned to SPEAKER_FRAMES_PER_DMA
    // by writing silence to the buffer.
    // Also write an extra buffer of silence to we can properly stop the DMA circular buffer before it's too late.
    if(speaker.n_mp3_bytes_remaining <= 0)
    {
      const int rounded_samples_length = ALIGN_UP(speaker.n_processed_pcm_samples, SPEAKER_FRAMES_PER_DMA);
      const int padding_samples = (rounded_samples_length - speaker.n_processed_pcm_samples) + SPEAKER_FRAMES_PER_DMA;
      n_frames_processed += padding_samples;
      for(int i = padding_samples; i > 0; --i)
      {
        vdac_ptr = write_vdac_buffer(vdac_ptr, 0);
      }
    }

    speaker.vdac_buffer_ptr = vdac_ptr;
    CORE_ATOMIC_SECTION(
      speaker.n_buffered_pcm_samples += n_frames_processed;
    );
  }

  // If there's any data remaining in the local MP3 buffer
  // then move it so the beginning of the buffer
  // so it'll be processed first next time the function is called
  if(mp3_buffer_remaining > 0 && mp3_buffer_remaining != speaker.mp3_buffer_used)
  {
    memcpy(speaker.mp3_buffer, mp3_ptr, mp3_buffer_remaining);
  }
  speaker.mp3_buffer_used = mp3_buffer_remaining;

  if(speaker.n_buffered_pcm_samples >= SPEAKER_FRAMES_PER_DMA)
  {
    // Start audio playback if necessary
    speaker_start_playback();
  }
}