#include "em_device.h"
#include "em_cmu.h"
#include "em_gpio.h"
#include "em_i2c.h"
#include "em_usart.h"
#include "dmadrv.h"


#include "arducam/arducam.h"
#include "arducam_m_2mp_driver.h"


#define SL_CONCAT(A, B, C) A ## B ## C
/* Generate the cmu clock symbol based on instance. */
#define I2C_CLOCK(N) SL_CONCAT(cmuClock_I2C, N, )
#define I2C_PORT(N) SL_CONCAT(I2C, N, )

#define USART_CLOCK(N) SL_CONCAT(cmuClock_USART, N, )
#define USART_PORT(N) SL_CONCAT(USART, N, )
#define USART_RX_DMA_SIGNAL(N) SL_CONCAT(dmadrvPeripheralSignal_USART, N, _RXDATAV)
#define USART_TX_DMA_SIGNAL(N) SL_CONCAT(dmadrvPeripheralSignal_USART, N, _TXBL)

#if defined(_SILICON_LABS_32B_SERIES_2)
#define ENABLE_SPI_INTERRUPT() LDMA->IEN_SET = (1 << arducam_context.dma_rx_channel)
#define DISABLE_SPI_INTERRUPT() LDMA->IEN_CLR = (1 << arducam_context.dma_rx_channel)
#else 
#define ENABLE_SPI_INTERRUPT() LDMA->IEN |= (1 << arducam_context.dma_rx_channel)
#define DISABLE_SPI_INTERRUPT() LDMA->IEN &= ~(1 << arducam_context.dma_rx_channel)
#endif



/*************************************************************************************************/
sl_status_t arducam_driver_init(const arducam_config_t* config)
{
    sl_status_t status;

    CMU_ClockEnable(cmuClock_GPIO, true);
#if defined(_CMU_HFPERCLKEN0_MASK)
    CMU_ClockEnable(cmuClock_HFPER, true);
#endif

    // ---------------------------------------------------
    // Initialize SPI interface
    // ---------------------------------------------------
    CMU_ClockEnable(USART_CLOCK(ARDUCAM_USART_PERIPHERAL_NO), true);

    // Configure GPIO mode
    GPIO_PinModeSet(ARDUCAM_USART_CLK_PORT, ARDUCAM_USART_CLK_PIN, gpioModePushPull, 0); // US0_CLK is push pull
    GPIO_PinModeSet(ARDUCAM_USART_CS_PORT,  ARDUCAM_USART_CS_PIN,  gpioModePushPull, 1); // US0_CS is push pull
    GPIO_PinModeSet(ARDUCAM_USART_TX_PORT,  ARDUCAM_USART_TX_PIN,  gpioModePushPull, 1); // US0_TX (MOSI) is push pull
    GPIO_PinModeSet(ARDUCAM_USART_RX_PORT,  ARDUCAM_USART_RX_PIN,  gpioModeInput, 1);    // US0_RX (MISO) is input

    USART_InitSync_TypeDef usart_config = USART_INITSYNC_DEFAULT;
    usart_config.master       = true;            // master mode
    usart_config.baudrate     = 1000000;         // CLK freq is 1MHz 
                                                 // NOTE: This can technically go up to 8M but can be unreliable depending on the hardware setup
    usart_config.autoCsEnable = true;            // CS pin controlled by hardware, not firmware
    usart_config.clockMode    = usartClockMode0; // clock idle low, sample on rising/first edge
    usart_config.msbf         = true;            // send MSB first
    usart_config.enable       = usartDisable;    // making sure USART isn't enabled until we set it up
    USART_InitSync(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO), &usart_config);

    // Set USART pin locations
#if defined(_SILICON_LABS_32B_SERIES_2)
    GPIO->USARTROUTE[ARDUCAM_USART_PERIPHERAL_NO].ROUTEEN = GPIO_USART_ROUTEEN_TXPEN
                                           | GPIO_USART_ROUTEEN_RXPEN
                                           | GPIO_USART_ROUTEEN_CLKPEN;

    GPIO->USARTROUTE[ARDUCAM_USART_PERIPHERAL_NO].TXROUTE = ((uint32_t)ARDUCAM_USART_TX_PORT
                                            << _GPIO_USART_TXROUTE_PORT_SHIFT)
                                           | ((uint32_t)ARDUCAM_USART_TX_PIN
                                              << _GPIO_USART_TXROUTE_PIN_SHIFT);

    GPIO->USARTROUTE[ARDUCAM_USART_PERIPHERAL_NO].RXROUTE = ((uint32_t)ARDUCAM_USART_RX_PORT
                                            << _GPIO_USART_RXROUTE_PORT_SHIFT)
                                           | ((uint32_t)ARDUCAM_USART_RX_PIN
                                              << _GPIO_USART_RXROUTE_PIN_SHIFT);

    GPIO->USARTROUTE[ARDUCAM_USART_PERIPHERAL_NO].CLKROUTE = ((uint32_t)ARDUCAM_USART_CLK_PORT
                                             << _GPIO_USART_CLKROUTE_PORT_SHIFT)
                                            | ((uint32_t)ARDUCAM_USART_CLK_PIN
                                               << _GPIO_USART_CLKROUTE_PIN_SHIFT);

#else
    USART_PORT(ARDUCAM_USART_PERIPHERAL_NO)->ROUTELOC0 = (ARDUCAM_USART_CLK_LOC) |
                        (ARDUCAM_USART_TX_LOC)  |
                        (ARDUCAM_USART_RX_LOC); 

    // Enable USART pins
    USART_PORT(ARDUCAM_USART_PERIPHERAL_NO)->ROUTEPEN = USART_ROUTEPEN_CLKPEN | USART_ROUTEPEN_TXPEN | USART_ROUTEPEN_RXPEN;
#endif 

    // Enable USART
    USART_Enable(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO), usartEnable);

    // Allocate a DMA channel for burst reads
    DMADRV_Init();
    if(DMADRV_AllocateChannel(&arducam_context.dma_rx_channel, NULL) != 0)
    {
        return SL_STATUS_NO_MORE_RESOURCE;
    }
    if(DMADRV_AllocateChannel(&arducam_context.dma_tx_channel, NULL) != 0)
    {
        return SL_STATUS_NO_MORE_RESOURCE;
    }


    // ---------------------------------------------------
    // Initialize I2C interface
    // ---------------------------------------------------
    CMU_ClockEnable(I2C_CLOCK(ARDUCAM_I2C_PERIPHERAL_NO), true);

    /* Output value must be set to 1 to not drive lines low. Set
     SCL first, to ensure it is high before changing SDA. */
    GPIO_PinModeSet(ARDUCAM_I2C_SCL_PORT, ARDUCAM_I2C_SCL_PIN, gpioModeWiredAndPullUp, 1);
    GPIO_PinModeSet(ARDUCAM_I2C_SDA_PORT, ARDUCAM_I2C_SDA_PIN, gpioModeWiredAndPullUp, 1);
   
    /* In some situations, after a reset during an I2C transfer, the slave
        device may be left in an unknown state. Send 9 clock pulses to
        set slave in a defined state. */
    for (int i = 0; i < 9; i++) 
    {
        GPIO_PinOutClear(ARDUCAM_I2C_SCL_PORT, ARDUCAM_I2C_SCL_PIN);
        ARDUCAM_DELAY_US(100);
        GPIO_PinOutSet(ARDUCAM_I2C_SCL_PORT, ARDUCAM_I2C_SDA_PIN);
        ARDUCAM_DELAY_US(100);
    }


  /* Enable pins and set location */
#if defined(_SILICON_LABS_32B_SERIES_2)
    GPIO->I2CROUTE[ARDUCAM_I2C_PERIPHERAL_NO].ROUTEEN = GPIO_I2C_ROUTEEN_SDAPEN | GPIO_I2C_ROUTEEN_SCLPEN;
    GPIO->I2CROUTE[ARDUCAM_I2C_PERIPHERAL_NO].SCLROUTE = 
        (uint32_t)((ARDUCAM_I2C_SCL_PIN << _GPIO_I2C_SCLROUTE_PIN_SHIFT) | (ARDUCAM_I2C_SCL_PORT << _GPIO_I2C_SCLROUTE_PORT_SHIFT));
    GPIO->I2CROUTE[ARDUCAM_I2C_PERIPHERAL_NO].SDAROUTE = 
        (uint32_t)((ARDUCAM_I2C_SDA_PIN << _GPIO_I2C_SDAROUTE_PIN_SHIFT) | (ARDUCAM_I2C_SDA_PORT << _GPIO_I2C_SDAROUTE_PORT_SHIFT));

#else
    I2C_PORT(ARDUCAM_I2C_PERIPHERAL_NO)->ROUTEPEN  = I2C_ROUTEPEN_SDAPEN | I2C_ROUTEPEN_SCLPEN;
    I2C_PORT(ARDUCAM_I2C_PERIPHERAL_NO)->ROUTELOC0 = (uint32_t)((ARDUCAM_I2C_SDA_LOC << _I2C_ROUTELOC0_SDALOC_SHIFT)
                                     | (ARDUCAM_I2C_SCL_LOC << _I2C_ROUTELOC0_SCLLOC_SHIFT));
#endif

    I2C_Init_TypeDef i2c_config = I2C_INIT_DEFAULT;

    i2c_config.enable = true;
    i2c_config.master = true; /* master mode only */
    i2c_config.freq = I2C_FREQ_STANDARD_MAX;
    i2c_config.refFreq = 0;
    i2c_config.clhr = i2cClockHLRStandard;

    I2C_Init(I2C_PORT(ARDUCAM_I2C_PERIPHERAL_NO), &i2c_config);


    // ---------------------------------------------------
    // Initialize the OV2640 
    // ---------------------------------------------------
    status = ov2640_init(config);
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    // ---------------------------------------------------
    // Configure the camera FIFO registers
    // ---------------------------------------------------
#define SPI_TEST_VALUE 0xBA
    uint8_t reg_value;

    reg_value = 0;
    arducam_driver_spi_read_reg(ARDUCHIP_REG_REV, &reg_value);


    if(reg_value != 0x73)
    {
        // Older Arducam versions do not require a dummy byte between the burst read command and RX data
        arducam_context.add_dummy_byte_to_burst_read = true;
    }

    reg_value = 0;
    arducam_driver_spi_write_reg(ARDUCHIP_REG_TEST1, SPI_TEST_VALUE);
    arducam_driver_spi_read_reg(ARDUCHIP_REG_TEST1, &reg_value);

    if(reg_value != SPI_TEST_VALUE)
    {
        return SL_STATUS_BUS_ERROR;
    }

    arducam_driver_spi_write_reg(ARDUCHIP_REG_FIFO, FIFO_CLEAR_MASK|FIFO_RDPTR_RST_MASK|FIFO_WRPTR_RST_MASK);
    arducam_driver_spi_set_bit(ARDUCHIP_REG_TIM, MODE_MASK);
    arducam_driver_spi_write_reg(ARDUCHIP_REG_FRAMES, 1);
    arducam_driver_spi_read_reg(ARDUCHIP_REG_FRAMES, &reg_value);

    if(reg_value != 1)
    {
        return SL_STATUS_BUS_ERROR;
    }

    return SL_STATUS_OK;
}

/*************************************************************************************************/
sl_status_t arducam_driver_deinit()
{
    ov2640_deinit();
    I2C_Reset(I2C_PORT(ARDUCAM_I2C_PERIPHERAL_NO));
    USART_Reset(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO));

    CMU_ClockEnable(I2C_CLOCK(ARDUCAM_I2C_PERIPHERAL_NO), false);
    CMU_ClockEnable(USART_CLOCK(ARDUCAM_USART_PERIPHERAL_NO), false);

    if(arducam_context.dma_rx_channel != -1)
    {
        DMADRV_FreeChannel(arducam_context.dma_rx_channel);
        arducam_context.dma_rx_channel = -1;
    }

    if(arducam_context.dma_tx_channel != -1)
    {
        DMADRV_FreeChannel(arducam_context.dma_tx_channel);
        arducam_context.dma_tx_channel = -1;
    }

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static sl_status_t do_i2c_transfer(I2C_TransferSeq_TypeDef *seq)
{
    I2C_TransferReturn_TypeDef ret;
    uint32_t timeout = 300000;

    /* Do a polled transfer */
    ret = I2C_TransferInit(I2C_PORT(ARDUCAM_I2C_PERIPHERAL_NO), seq);
    while (ret == i2cTransferInProgress && timeout--) 
    {
        ret = I2C_Transfer(I2C_PORT(ARDUCAM_I2C_PERIPHERAL_NO));
    }

  return (ret == i2cTransferDone) ? SL_STATUS_OK : SL_STATUS_BUS_ERROR;
}

/*************************************************************************************************/
sl_status_t arducam_driver_i2c_write_reg(uint8_t addr, uint8_t data)
{
    I2C_TransferSeq_TypeDef seq;
    I2C_TransferReturn_TypeDef ret;
    uint8_t write_buffer[2] = {addr, data};

    seq.addr = OV2640_I2C_ADDRESS << 1;
    seq.flags = I2C_FLAG_WRITE;

    seq.buf[0].data = write_buffer;
    seq.buf[0].len  = 2;

    return do_i2c_transfer(&seq);
}

/*************************************************************************************************/
sl_status_t arducam_driver_i2c_read_reg(uint8_t addr, uint8_t *val)
{
    I2C_TransferSeq_TypeDef seq;
    I2C_TransferReturn_TypeDef ret;

    seq.addr = OV2640_I2C_ADDRESS << 1;
    seq.flags = I2C_FLAG_WRITE_READ;

    seq.buf[0].data = &addr;
    seq.buf[0].len  = 1;

    seq.buf[1].data = val;
    seq.buf[1].len  = 1;

    return do_i2c_transfer(&seq);
}

/*************************************************************************************************/
sl_status_t arducam_driver_i2c_write_regs(
    const reg_addr_value_t *regs, 
    const reg_addr_value_t *action_list,
    uint8_t action_list_len
)
{
    for(const reg_addr_value_t *addr_value = regs;; ++addr_value)
    {
        if(addr_value->address == REG_ADDR_ACTION)
        {
            if(addr_value->value == REG_ACTION_TERMINATE)
            {
                return SL_STATUS_OK;
            }
            else if(action_list == NULL)
            {
                return SL_STATUS_NULL_POINTER;
            }
            else if(addr_value->value >= action_list_len)
            {
                return SL_STATUS_HAS_OVERFLOWED;
            }
            else
            {
                const reg_addr_value_t *action = &action_list[addr_value->value];
                SL_VERIFY(arducam_driver_i2c_write_reg(action->address, action->value));
            }
        }
        else
        {
            SL_VERIFY(arducam_driver_i2c_write_reg(addr_value->address, addr_value->value));
        }
    }

    return SL_STATUS_OK;
}

/*************************************************************************************************/
sl_status_t arducam_driver_spi_write_reg(uint8_t addr, uint8_t data)
{
    DISABLE_SPI_INTERRUPT();

    if(arducam_context.is_spi_active)
    {
        ENABLE_SPI_INTERRUPT();
        return SL_STATUS_INVALID_STATE;
    }

    arducam_context.is_spi_active = true;

    GPIO_PinOutClear(ARDUCAM_USART_CS_PORT,  ARDUCAM_USART_CS_PIN);

    //ARDUCAM_DELAY_US(5);

    USART_SpiTransfer(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO), addr | ARDUCHIP_RW_FLAG);
    USART_SpiTransfer(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO), data);

    //ARDUCAM_DELAY_US(5);

    GPIO_PinOutSet(ARDUCAM_USART_CS_PORT,  ARDUCAM_USART_CS_PIN);

    ARDUCAM_DEBUG("SPI TX: 0x%02X = 0x%02X", addr, data);

    arducam_context.is_spi_active = false;

    ENABLE_SPI_INTERRUPT();

    return SL_STATUS_OK;
}

/*************************************************************************************************/
sl_status_t arducam_driver_spi_read_reg(uint8_t addr, uint8_t *data_ptr)
{
    DISABLE_SPI_INTERRUPT();

    if(arducam_context.is_spi_active)
    {
        ENABLE_SPI_INTERRUPT();
        return SL_STATUS_INVALID_STATE;
    }

    arducam_context.is_spi_active = true;

    GPIO_PinOutClear(ARDUCAM_USART_CS_PORT,  ARDUCAM_USART_CS_PIN);

    //ARDUCAM_DELAY_US(5);

    USART_SpiTransfer(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO), addr);
    *data_ptr = USART_SpiTransfer(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO), 0xFF);

    //ARDUCAM_DELAY_US(5);

    GPIO_PinOutSet(ARDUCAM_USART_CS_PORT,  ARDUCAM_USART_CS_PIN);

    //if(addr !=ARDUCHIP_REG_STATUS || (*data_ptr & CAP_DONE_MASK) != 0)
    {
        ARDUCAM_DEBUG("SPI RX: 0x%02X = 0x%02X", addr, *data_ptr);
    }
   

    arducam_context.is_spi_active = false;

    ENABLE_SPI_INTERRUPT();

    return SL_STATUS_OK;
}


/*************************************************************************************************/
sl_status_t arducam_driver_spi_clear_bit(uint8_t addr, uint8_t bits)
{
    uint8_t reg_value;

    SL_VERIFY(arducam_driver_spi_read_reg(addr, &reg_value));

    reg_value &= ~bits;

    SL_VERIFY(arducam_driver_spi_write_reg(addr, reg_value));

    return SL_STATUS_OK;
}

/*************************************************************************************************/
sl_status_t arducam_driver_spi_set_bit(uint8_t addr, uint8_t bits)
{
    uint8_t reg_value;

    SL_VERIFY(arducam_driver_spi_read_reg(addr, &reg_value));

    reg_value |= bits;

    SL_VERIFY(arducam_driver_spi_write_reg(addr, reg_value));

    return SL_STATUS_OK;
}

/*************************************************************************************************/
static bool on_dma_rx_complete_irq_handler(
    unsigned int channel,
    unsigned int sequenceNo,
    void *userParam
)
{
#define MAX_DMA_LENGTH ((_LDMA_CH_CTRL_XFERCNT_MASK >> _LDMA_CH_CTRL_XFERCNT_SHIFT)+1)
    static uint8_t dummy_tx_buffer = 0;

    if(userParam == NULL)
    {
        arducam_context.dma_rx_ptr += MAX_DMA_LENGTH;
        arducam_context.dma_length_remaining -= MIN(MAX_DMA_LENGTH, arducam_context.dma_length_remaining);
    }

    if(arducam_context.dma_length_remaining == 0)
    {
        GPIO_PinOutSet(ARDUCAM_USART_CS_PORT, ARDUCAM_USART_CS_PIN);
        arducam_context.state = CAMERA_STATE_READ_COMPLETE;
        arducam_context.is_spi_active = false;
        arducam_poll();
        return false;
    }


    const uint32_t chunk_length = MIN(MAX_DMA_LENGTH, arducam_context.dma_length_remaining);

    LDMA_TransferCfg_t rx_cfg = LDMA_TRANSFER_CFG_PERIPHERAL(USART_RX_DMA_SIGNAL(ARDUCAM_USART_PERIPHERAL_NO));
    LDMA_Descriptor_t rx_desc = LDMA_DESCRIPTOR_LINKREL_P2M_BYTE(
        &USART_PORT(ARDUCAM_USART_PERIPHERAL_NO)->RXDATA,
        arducam_context.dma_rx_ptr,
        chunk_length,
        0
    );
    rx_desc.xfer.link = 0;
    rx_desc.xfer.doneIfs = 1;


    LDMA_TransferCfg_t tx_cfg = LDMA_TRANSFER_CFG_PERIPHERAL(USART_TX_DMA_SIGNAL(ARDUCAM_USART_PERIPHERAL_NO));
    LDMA_Descriptor_t tx_desc = LDMA_DESCRIPTOR_LINKREL_M2P_BYTE(
        &dummy_tx_buffer,
        &USART_PORT(ARDUCAM_USART_PERIPHERAL_NO)->TXDATA,
        chunk_length,
        0
    );
    tx_desc.xfer.link = 0;
    tx_desc.xfer.doneIfs = 0;
    tx_desc.xfer.srcInc = ldmaCtrlSrcIncNone;

    // Start receive DMA
    DMADRV_LdmaStartTransfer(
        arducam_context.dma_rx_channel, &rx_cfg,
        &rx_desc, on_dma_rx_complete_irq_handler, NULL
    );
    
    // Start transmit DMA
    DMADRV_LdmaStartTransfer(
        arducam_context.dma_tx_channel, &tx_cfg,
        &tx_desc, NULL, NULL
    );

    return false;
}


/*************************************************************************************************/
sl_status_t arducam_driver_spi_burst_read(uint8_t *buffer, uint32_t length)
{
    DISABLE_SPI_INTERRUPT();

    if(arducam_context.is_spi_active)
    {
        return SL_STATUS_INVALID_STATE;
    }

    // NOTE: The SPI interrupt will automatically be enabled
    arducam_context.is_spi_active = true;

    GPIO_PinOutClear(ARDUCAM_USART_CS_PORT, ARDUCAM_USART_CS_PIN);

    USART_SpiTransfer(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO), BURST_FIFO_READ);

    if(arducam_context.add_dummy_byte_to_burst_read)
    {
        USART_SpiTransfer(USART_PORT(ARDUCAM_USART_PERIPHERAL_NO), 0x00);
    }

    arducam_context.dma_length_remaining = length;
    arducam_context.dma_rx_ptr = buffer;
    on_dma_rx_complete_irq_handler(0, 0, (void*)1);


    return SL_STATUS_OK;
}

/*************************************************************************************************/
sl_status_t arducam_driver_get_fifo_size(uint32_t *size_ptr)
{
    uint32_t size;
    sl_status_t status;
    uint8_t size_part;

    *size_ptr = 0;

    status = arducam_driver_spi_read_reg(FIFO_SIZE3, &size_part);
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    size = size_part;
    size <<= 8;

    status = arducam_driver_spi_read_reg(FIFO_SIZE2, &size_part);
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    size |= (uint32_t)size_part;
    size <<= 8;

    status = arducam_driver_spi_read_reg(FIFO_SIZE1, &size_part);
    if(status != SL_STATUS_OK)
    {
        return status;
    }

    size |= (uint32_t)size_part;

    if(size > MAX_IMAGE_SIZE)
    {
        return SL_STATUS_BUS_ERROR;
    }

    *size_ptr = size;

    return SL_STATUS_OK;
}
