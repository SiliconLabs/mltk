#include <cstdio>
#include <cstdlib>
#include <cassert>

#include "em_device.h"
#include "em_chip.h"
#include "em_cmu.h"
#include "em_emu.h"
#include "em_vdac.h"
#include "em_timer.h"
#include "em_ldma.h"
#include "em_core.h"
#include "dmadrv.h"

#include "sl_system_init.h"



#define MINIMP3_ONLY_MP3
#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"

#include "test_mp3_data.h"

#define MAX_XFR_PER_DMA 2048
#define MPEG2_FRAME_LENGTH 576
#define SPEAKER_FRAMES_PER_DMA ((MAX_XFR_PER_DMA/MPEG2_FRAME_LENGTH)*MPEG2_FRAME_LENGTH)
#define SPEAKER_DMA_COUNT 3

#define VDAC_CHANNEL_NUM                 0
// Set the VDAC to max frequency of 1 MHz
#define VDAC_CLOCK_FREQ                   1000000


static void init_vdac(void);
static void init_timer(int);
static void init_ldma(void);
static void convert_pcm_to_vdac(const int16_t* pcm_samples, int n_samples, int channels);
static void start_playback(bool start);



static mp3dec_t mp3d;


static struct
{
  uint16_t vdac_buffer[SPEAKER_DMA_COUNT][SPEAKER_FRAMES_PER_DMA];
  uint16_t* vdac_buffer_end;
  LDMA_Descriptor_t dma_descriptors[SPEAKER_DMA_COUNT];
  uint16_t* vdac_buffer_ptr;
  volatile int n_frames_pending;
  bool started;
} speaker;



extern "C" int main(void)
{
    sl_system_init();

    printf("Mini MP3 example\n");

    speaker.vdac_buffer_ptr = speaker.vdac_buffer[0];
    speaker.vdac_buffer_end = &speaker.vdac_buffer_ptr[SPEAKER_DMA_COUNT*SPEAKER_FRAMES_PER_DMA];
    speaker.n_frames_pending = 0;


    mp3dec_init(&mp3d);

    const uint8_t *mp3_ptr = MP3_DATA;
    int data_remaining = MP3_DATA_LENGTH;
    mp3dec_frame_info_t info;
    int16_t pcm_samples[MINIMP3_MAX_SAMPLES_PER_FRAME];

    int n_samples = mp3dec_decode_frame(&mp3d, mp3_ptr, data_remaining, pcm_samples, &info);
    convert_pcm_to_vdac(pcm_samples, n_samples, info.channels);
    mp3_ptr += info.frame_bytes;
    data_remaining -= info.frame_bytes;

    init_vdac();
    init_ldma();
    init_timer(info.hz);
    

    while(data_remaining > 0)
    {
      while(speaker.n_frames_pending >= SPEAKER_FRAMES_PER_DMA*(SPEAKER_DMA_COUNT-1))
      {
        EMU_EnterEM1();
      }

      int n_samples = mp3dec_decode_frame(&mp3d, mp3_ptr, data_remaining, pcm_samples, &info);
      assert(n_samples > 0);
      mp3_ptr += info.frame_bytes;
      data_remaining -= info.frame_bytes;
      convert_pcm_to_vdac(pcm_samples, n_samples, info.channels);
      start_playback(true);
    }

    while(speaker.n_frames_pending > 0)
    {
      EMU_EnterEM1();
    }

    return 0;
}

/*************************************************************************************************/
static void convert_pcm_to_vdac(const int16_t* pcm_samples, int n_samples, int channels)
{
    const int16_t *src = pcm_samples;
    uint16_t* dst = speaker.vdac_buffer_ptr;

    for(int i = n_samples; i > 0; --i)
    {
      *dst++ = (uint16_t)(((int32_t)*src + INT16_MAX) >> 4);
      src += channels;
      if(dst >= speaker.vdac_buffer_end)
      {
        dst = speaker.vdac_buffer[0];
      }
    }
    speaker.vdac_buffer_ptr = dst;
    speaker.n_frames_pending += n_samples;
}

/*************************************************************************************************/
static void start_playback(bool start)
{
  if(start)
  {
    if(!speaker.started)
    {
      speaker.started = true;
      TIMER_Enable(TIMER0, true);
    }
  }
  else 
  {
      speaker.started = false;
      TIMER_Enable(TIMER0, false);
  }
}

/*************************************************************************************************/
void init_vdac(void)
{
  // Use default settings
  VDAC_Init_TypeDef        init        = VDAC_INIT_DEFAULT;
  VDAC_InitChannel_TypeDef initChannel = VDAC_INITCHANNEL_DEFAULT;

  // The EM01GRPACLK is chosen as VDAC clock source since the VDAC will be
  // operating in EM1
  CMU_ClockSelectSet(cmuClock_VDAC0, cmuSelect_EM01GRPACLK);

  // Enable the VDAC clocks
  CMU_ClockEnable(cmuClock_VDAC0, true);

  // Calculate the VDAC clock prescaler value resulting in a 1 MHz VDAC clock.
  init.prescaler = VDAC_PrescaleCalc(VDAC0, (uint32_t)VDAC_CLOCK_FREQ);

  // Set reference to internal 1.25V low noise reference
  init.reference = vdacRef2V5;

  // Since the minimum load requirement for high capacitance mode is 25 nF, turn
  // this mode off
  initChannel.highCapLoadEnable = false;
  initChannel.powerMode = vdacPowerModeHighPower;

  // Initialize the VDAC and VDAC channel
  VDAC_Init(VDAC0, &init);
  VDAC_InitChannel(VDAC0, &initChannel, VDAC_CHANNEL_NUM);

  // Enable the VDAC
  VDAC_Enable(VDAC0, VDAC_CHANNEL_NUM, true);
}


/*************************************************************************************************/
void init_timer(int sample_rate)
{
  uint32_t timerFreq, topValue;

  // Enable clock for TIMER0 module
  CMU_ClockEnable(cmuClock_TIMER0, true);

  // Initialize TIMER0
  TIMER_Init_TypeDef init = TIMER_INIT_DEFAULT;
  init.dmaClrAct = true;
  init.enable = false;
  TIMER_Init(TIMER0, &init);

  // Set top (reload) value for the timer
  // Note: the timer runs off of the EM01GRPACLK clock
  timerFreq = CMU_ClockFreqGet(cmuClock_TIMER0) / (init.prescale + 1);
  topValue = (timerFreq / sample_rate);

  // Set top value to overflow at the desired TIMER0_FREQ frequency
  TIMER_TopSet(TIMER0, topValue);
}


/*************************************************************************************************/
static bool vdac_dma_callback(unsigned int channel, unsigned int sequenceNo, void *userParam)
{
  speaker.n_frames_pending -= SPEAKER_FRAMES_PER_DMA;
  if(speaker.n_frames_pending <= 0)
  {
    start_playback(false);
  }

  return true;
}


/*************************************************************************************************/
void init_ldma(void)
{
  static const LDMA_Descriptor_t desc_template = {
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
      .dstAddr      = (uint32_t)&VDAC0->CH0F,
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
    LDMA_TRANSFER_CFG_PERIPHERAL(ldmaPeripheralSignal_TIMER0_UFOF);

  DMADRV_Init();
  unsigned int ch;
  DMADRV_AllocateChannel(&ch, nullptr);

  DMADRV_LdmaStartTransfer(ch, &transferConfig, speaker.dma_descriptors, vdac_dma_callback, nullptr);
}