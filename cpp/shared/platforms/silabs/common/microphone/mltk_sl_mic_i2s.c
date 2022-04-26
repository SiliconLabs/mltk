/***************************************************************************//**
 * @file
 * @brief I2S microphone driver
 *******************************************************************************
 * # License
 * <b>Copyright 2020 Silicon Laboratories Inc. www.silabs.com</b>
 *******************************************************************************
 *
 * SPDX-License-Identifier: Zlib
 *
 * The licensor of this software is Silicon Laboratories Inc.
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 *
 ******************************************************************************/
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "em_cmu.h"
#include "em_usart.h"
#include "em_ldma.h"
#include "dmadrv.h"
#include "mltk_sl_mic.h"
#include "sl_mic_i2s_config.h"


#define MAX_DMA_LENGTH 2048

/* Concatenate preprocessor tokens A, B and C. */
#define SL_CONCAT(A, B, C) A ## B ## C
/* Generate the cmu clock symbol based on instance. */
#define MIC_I2S_USART_CLOCK(N) SL_CONCAT(cmuClock_USART, N, )
#define MIC_DMA_LEFT_SIGNAL(N) SL_CONCAT(ldmaPeripheralSignal_USART, N, _RXDATAV)
#define MIC_DMA_RIGHT_SIGNAL(N) SL_CONCAT(ldmaPeripheralSignal_USART, N, _RXDATAVRIGHT)

#ifndef DIV_ROUND_UP
#define DIV_ROUND_UP(m, n)    (((m) + (n) - 1) / (n))
#endif /* ifndef DIV_ROUND_UP */


static struct
{
    uint32_t                        sample_rate;
    uint8_t                         n_channels;
    uint32_t                        n_frames_per_callback;         
    sl_mic_buffer_ready_callback_t  callback;

    struct
    {
      int16_t*                      base;
      int16_t*                      end;
      int16_t*                      ptr;
      LDMA_Descriptor_t*            dma_desc;
      unsigned int                  dma_ch;
    } buffer;

    struct
    {
        LDMA_Descriptor_t           dma_desc;
        int16_t                     buffer;
        unsigned int                dma_ch;
    } __ALIGNED(4) drop_channel;



    bool initialized;
    bool is_streaming;

} mic_context = 
{
  .sample_rate = 0,
  .n_channels = 0,
  .n_frames_per_callback = 0,
  .callback = NULL,
  .buffer.base = NULL,
  .buffer.end = NULL,
  .buffer.ptr = NULL,
  .buffer.dma_desc = NULL,
  .buffer.dma_ch = EMDRV_DMADRV_DMA_CH_COUNT,
  .drop_channel.dma_ch = EMDRV_DMADRV_DMA_CH_COUNT,
  .initialized = false,
  .is_streaming = false
};


static bool dma_complete(unsigned int channel, unsigned int sequenceNo, void *userParam);
static void init_dma_descriptors(uint32_t sample_count, uint32_t sample_length, bool have_callback);



/***************************************************************************//**
 *    Initializes the microphone
 ******************************************************************************/
sl_status_t mltk_sl_mic_init(uint32_t sample_rate, uint8_t n_channels)
{
  // Only one channel is supported
  if (n_channels < 1 || n_channels > 2) {
    return SL_STATUS_INVALID_PARAMETER;
  }
  if (mic_context.initialized) {
    return SL_STATUS_INVALID_STATE;
  }

  uint32_t status;
  USART_InitI2s_TypeDef usartInit = USART_INITI2S_DEFAULT;

  /* Enable clocks */
  CMU_ClockEnable(cmuClock_GPIO, true);
  CMU_ClockEnable(MIC_I2S_USART_CLOCK(SL_MIC_I2S_PERIPHERAL_NO), true);

  /* Setup GPIO pins */
  GPIO_PinModeSet(SL_MIC_I2S_RX_PORT, SL_MIC_I2S_RX_PIN, gpioModeInput, 0);
  GPIO_PinModeSet(SL_MIC_I2S_CLK_PORT, SL_MIC_I2S_CLK_PIN, gpioModePushPull, 0);
  GPIO_PinModeSet(SL_MIC_I2S_CS_PORT, SL_MIC_I2S_CS_PIN, gpioModePushPull, 0);

  /* Setup USART in I2S mode to get data from microphone */
  usartInit.sync.enable   = usartEnable;
  usartInit.sync.baudrate = sample_rate * 64; // 32-bit stereo frame
  usartInit.sync.autoTx   = true;
  usartInit.format        = usartI2sFormatW32D16;

  if (n_channels == 1) {
    // Split DMA requests to discard right-channel data
    usartInit.dmaSplit      = true;
  }

  USART_InitI2s(SL_MIC_I2S_PERIPHERAL, &usartInit);

#if defined(_SILICON_LABS_32B_SERIES_2)
  GPIO->USARTROUTE->ROUTEEN = GPIO_USART_ROUTEEN_RXPEN | GPIO_USART_ROUTEEN_CLKPEN | GPIO_USART_ROUTEEN_CSPEN;
  GPIO->USARTROUTE->RXROUTE = (SL_MIC_I2S_RX_PORT << _GPIO_USART_RXROUTE_PORT_SHIFT) | (SL_MIC_I2S_RX_PIN << _GPIO_USART_RXROUTE_PIN_SHIFT);
  GPIO->USARTROUTE->CLKROUTE = (SL_MIC_I2S_CLK_PORT << _GPIO_USART_CLKROUTE_PORT_SHIFT) | (SL_MIC_I2S_CLK_PIN << _GPIO_USART_CLKROUTE_PIN_SHIFT);
  GPIO->USARTROUTE->CSROUTE = (SL_MIC_I2S_CS_PORT << _GPIO_USART_CSROUTE_PORT_SHIFT) | (SL_MIC_I2S_CS_PIN << _GPIO_USART_CSROUTE_PIN_SHIFT);
#else
  SL_MIC_I2S_PERIPHERAL->ROUTELOC0 = (SL_MIC_I2S_RX_LOC << _USART_ROUTELOC0_RXLOC_SHIFT | SL_MIC_I2S_CLK_LOC << _USART_ROUTELOC0_CLKLOC_SHIFT | SL_MIC_I2S_CS_LOC << _USART_ROUTELOC0_CSLOC_SHIFT);
  SL_MIC_I2S_PERIPHERAL->ROUTEPEN  = (USART_ROUTEPEN_RXPEN | USART_ROUTEPEN_CLKPEN | USART_ROUTEPEN_CSPEN);
#endif

  /* Setup DMA driver to move samples from USART to memory */
  DMADRV_Init();

  /* Set up DMA channel for I2S left (microphone) data */
  status = DMADRV_AllocateChannel(&mic_context.buffer.dma_ch, NULL);
  if ( status != ECODE_EMDRV_DMADRV_OK ) {
    return SL_STATUS_FAIL;
  }


  if (n_channels == 1 ) {
    /* Set up DMA channel to discard I2S right data */
    status = DMADRV_AllocateChannel(&mic_context.drop_channel.dma_ch, NULL);
    if ( status != ECODE_EMDRV_DMADRV_OK ) {
      return SL_STATUS_FAIL;
    }
  }

  /* Driver parameters */
  mic_context.n_channels = n_channels;
  mic_context.sample_rate = sample_rate;
  mic_context.initialized = true;

  return SL_STATUS_OK;
}

/***************************************************************************//**
 *    De-initializes the microphone
 ******************************************************************************/
sl_status_t mltk_sl_mic_deinit(void)
{
  /* Stop sampling */
  mltk_sl_mic_stop();

  /* Reset USART peripheral and disable IO pins */
  USART_Reset(SL_MIC_I2S_PERIPHERAL);
  SL_MIC_I2S_PERIPHERAL->I2SCTRL = 0;

  GPIO_PinModeSet(SL_MIC_I2S_CLK_PORT, SL_MIC_I2S_CLK_PIN, gpioModeDisabled, 0);
  GPIO_PinModeSet(SL_MIC_I2S_RX_PORT, SL_MIC_I2S_RX_PIN, gpioModeDisabled, 0);
  GPIO_PinModeSet(SL_MIC_I2S_CS_PORT, SL_MIC_I2S_CS_PIN, gpioModeDisabled, 0);

  /* Free resources */
  DMADRV_FreeChannel(mic_context.buffer.dma_ch);
  mic_context.buffer.dma_ch = EMDRV_DMADRV_DMA_CH_COUNT;
  DMADRV_FreeChannel(mic_context.drop_channel.dma_ch);
  mic_context.drop_channel.dma_ch = EMDRV_DMADRV_DMA_CH_COUNT;

  mic_context.initialized = false;

  return SL_STATUS_OK;
}


/***************************************************************************//**
 *    Start streaming
 ******************************************************************************/
sl_status_t mltk_sl_mic_start_streaming(void *buffer, uint32_t n_frames, sl_mic_buffer_ready_callback_t callback)
{
  if (!mic_context.initialized) {
    return SL_STATUS_NOT_INITIALIZED;
  }

  if (mic_context.is_streaming) {
    return SL_STATUS_INVALID_STATE;
  }

  // Determine the number of ms between each DMA irq
  // This effectively determines the resolution of the callback
  // and thus how often the app processes the audio. 
  // Start at 25ms and increment until we find a multiple of the given n_frames
  const int ms_per_sample = (n_frames * 1000) / mic_context.sample_rate;
  uint32_t ms_per_callback;
  for(ms_per_callback = 25; ms_per_callback < ms_per_sample; ++ms_per_callback)
  {
    const int n_frames_per_callback = (ms_per_callback * mic_context.sample_rate) / 1000;

    if(n_frames % n_frames_per_callback == 0)
    {
      break;
    }
  }
  
  mic_context.callback = callback;
  mic_context.n_frames_per_callback = (ms_per_callback * mic_context.sample_rate) / 1000;

 
  const uint32_t sample_length = n_frames * mic_context.n_channels;
  const uint32_t callback_sample_length = mic_context.n_frames_per_callback * mic_context.n_channels;
  const uint32_t callbacks_per_sample = sample_length / callback_sample_length;
  // The DMA buffer can hold up to 2 samples
  const uint32_t callback_sample_count = callbacks_per_sample * 2;
  const uint32_t desc_per_callback_sample = DIV_ROUND_UP(callback_sample_length, MAX_DMA_LENGTH);
  const uint32_t alloc_size = sizeof(LDMA_Descriptor_t)*desc_per_callback_sample * callback_sample_count;

  mic_context.buffer.dma_desc = (LDMA_Descriptor_t*)malloc(alloc_size);
  if(mic_context.buffer.dma_desc == NULL)
  {
    return SL_STATUS_ALLOCATION_FAILED;
  }

  mic_context.buffer.base = (int16_t*)buffer;
  mic_context.buffer.ptr = mic_context.buffer.base;
  mic_context.buffer.end = mic_context.buffer.ptr + callback_sample_length*callback_sample_count;
  init_dma_descriptors(callback_sample_count, callback_sample_length, 1);

  LDMA_TransferCfg_t dma_config  = LDMA_TRANSFER_CFG_PERIPHERAL(MIC_DMA_LEFT_SIGNAL(SL_MIC_I2S_PERIPHERAL_NO));
  dma_config.ldmaDbgHalt = false;
  
  
  DMADRV_LdmaStartTransfer(mic_context.buffer.dma_ch,
                            &dma_config, mic_context.buffer.dma_desc,
                            dma_complete, NULL);

  if(mic_context.n_channels == 1)
  {
      LDMA_TransferCfg_t drop_dma_config  = LDMA_TRANSFER_CFG_PERIPHERAL(MIC_DMA_RIGHT_SIGNAL(SL_MIC_I2S_PERIPHERAL_NO));
      drop_dma_config.ldmaDbgHalt = false;
      DMADRV_LdmaStartTransfer(mic_context.drop_channel.dma_ch, 
                              &drop_dma_config, 
                              &mic_context.drop_channel.dma_desc,
                              NULL, NULL);
  }

  return SL_STATUS_OK;
}


/***************************************************************************//**
 *    Stops the microphone
 ******************************************************************************/
sl_status_t mltk_sl_mic_stop(void)
{
  /* Stop sampling */
  DMADRV_StopTransfer(mic_context.buffer.dma_ch);

  if (mic_context.n_channels == 1) {
    DMADRV_StopTransfer(mic_context.drop_channel.dma_ch);
  }

  if(mic_context.buffer.dma_desc != NULL) {
    free(mic_context.buffer.dma_desc);
    mic_context.buffer.dma_desc = NULL;
  }

  mic_context.is_streaming = false;
  
  return SL_STATUS_OK;
}




/** @cond DO_NOT_INCLUDE_WITH_DOXYGEN */

/***************************************************************************//**
 * @brief
 *    Called when the DMA complete interrupt fired
 *
 * @param[in] channel
 *    DMA channel
 *
 * @param[in] sequenceNo
 *    Sequence number
 *
 * @param[in] userParam
 *    User parameters
 *
 * @return
 *    Returns false to stop transfers
 ******************************************************************************/
static bool dma_complete(unsigned int channel, unsigned int sequenceNo, void *userParam)
{
  if(mic_context.callback != NULL)
  {
    int16_t* ptr = mic_context.buffer.ptr;
    mic_context.callback(ptr, mic_context.n_frames_per_callback);
    ptr += mic_context.n_frames_per_callback * mic_context.n_channels;
    if(ptr >= mic_context.buffer.end) {
      ptr = mic_context.buffer.base;
    }
    mic_context.buffer.ptr = ptr;
  }
  return true;
}

/***************************************************************************//**
 *   Initialize the DMA descriptors
 ******************************************************************************/
static void init_dma_descriptors(uint32_t sample_count, uint32_t sample_length, bool have_callback)
{
    static const LDMA_Descriptor_t desc_template =   {
      .xfer =
      {
        .structType   = ldmaCtrlStructTypeXfer,
        .structReq    = 0,
        .xferCnt      = 0,
        .byteSwap     = 0,
        .blockSize    = ldmaCtrlBlockSizeUnit1,
        .doneIfs      = 0, 
        .reqMode      = ldmaCtrlReqModeBlock,
        .decLoopCnt   = 0,
        .ignoreSrec   = 0,
        .srcInc       = ldmaCtrlSrcIncNone,
        .size         = ldmaCtrlSizeHalf,
        .dstInc       = ldmaCtrlDstIncOne,
        .srcAddrMode  = ldmaCtrlSrcAddrModeAbs,
        .dstAddrMode  = ldmaCtrlDstAddrModeAbs,
        .srcAddr      = (uint32_t)(&SL_MIC_I2S_PERIPHERAL->RXDOUBLE),
        .dstAddr      = (uint32_t)0,
        .linkMode     = ldmaLinkModeRel,
        .link         = 1,
        .linkAddr     = 4
      }
    };
    LDMA_Descriptor_t* desc_ptr = mic_context.buffer.dma_desc;
    int16_t *buffer_ptr = mic_context.buffer.base;
    int descriptor_count = 0;

    // For each sample buffer
    for(uint32_t sample_id = 0; sample_id < sample_count; ++sample_id)
    {
        // Populate the descriptors for the current sample buffer
        for(uint32_t length_remaining = sample_length; length_remaining > 0;)
        {
            const uint32_t chunk_length = (length_remaining > MAX_DMA_LENGTH) ?
                                          MAX_DMA_LENGTH : length_remaining;
            
            length_remaining -= chunk_length;

            *desc_ptr = desc_template;
            desc_ptr->xfer.xferCnt = chunk_length-1;
            desc_ptr->xfer.dstAddr = (uint32_t)buffer_ptr;
            // Only the last descriptor for this sample should generate an interrupt
            desc_ptr->xfer.doneIfs = (have_callback && (length_remaining == 0)) ? 1 : 0; 
            // Update the counters and pointers
            ++desc_ptr;
            ++descriptor_count;
            buffer_ptr += chunk_length;
        }
    }

    // The last DMA descriptor points to the first descriptor
    // to create a ring between the sample buffers:
    // buffer1 -> buffer2 -> buffer3 -> buffer1 -> ...
    --desc_ptr;
    desc_ptr->xfer.linkAddr = -((descriptor_count-1)*4);

    if(mic_context.n_channels == 1)
    {
        mic_context.drop_channel.dma_desc  = desc_template;
        mic_context.drop_channel.dma_desc.xfer.dstAddr = (uint32_t)&mic_context.drop_channel.buffer;
        mic_context.drop_channel.dma_desc.xfer.linkAddr = 0;
        mic_context.drop_channel.dma_desc.xfer.doneIfs = 0;
        mic_context.drop_channel.dma_desc.xfer.dstInc = ldmaCtrlDstIncNone;
    }
}

