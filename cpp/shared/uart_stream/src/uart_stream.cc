#include "uart_stream.hpp"

#include <algorithm>

#include <cstdlib>
#include <cstring>
#include <cassert>

#include "em_core.h"
#include "em_emu.h"
#include "em_ldma.h"
#include "dmadrv.h"

#include "uart_stream_internal.h"


#define MAX_DMA_LENGTH 2048


namespace uart_stream
{



static int8_t write_data_packet_header(int32_t tx_length, bool block = true);
static bool start_next_rx_dma_transfer(unsigned int unused=0,unsigned int unused2=0, void *unused3=nullptr);
static bool start_next_tx_dma_transfer(unsigned int unused=0,unsigned int unused2=0, void *unused3=nullptr);
static void wait_for_tx_dma_to_complete();



static struct
{
    UartStreamDmaConfig dma_config;

    struct
    {
        int32_t buffer_length;
        uint8_t* head;
        uint8_t* tail;
        uint8_t* start;
        uint8_t* end;
        uint8_t header_buffer[sizeof(PacketHeader)];
        volatile int32_t bytes_available;
        volatile int32_t dma_bytes_remaining;
        unsigned int dma_channel;
        int32_t dma_chunk_length;
        volatile int8_t previous_data_packet_id;
        volatile bool cmd_available;
        uint8_t cmd_code;
        uint8_t cmd_payload[COMMAND_PAYLOAD_LENGTH];
    } rx;

    struct
    {
        volatile int32_t bytes_available;
        volatile int8_t active_data_packet_id;
        int8_t next_packet_id;
        unsigned int dma_channel;
        volatile int32_t dma_bytes_remaining;
        int32_t dma_chunk_length;
        const uint8_t* dma_ptr;
        volatile bool send_ack_after_transfer;
    } tx;

    volatile bool sync_requested;

} context;





/*************************************************************************************************/
bool initialize(
    uint8_t* rx_buffer,
    uint32_t rx_buffer_length,
    uint32_t baud_rate
)
{
    memset(&context, 0, sizeof(context));

    if(!uart_stream_internal_init(baud_rate))
    {
        return false;
    }

    DMADRV_Init();

    if(DMADRV_AllocateChannel(&context.tx.dma_channel, nullptr) != 0)
    {
        return false;
    }

    if(DMADRV_AllocateChannel(&context.rx.dma_channel, nullptr) != 0)
    {
        return false;
    }

    uart_stream_internal_get_dma_config(&context.dma_config);

    context.rx.buffer_length = rx_buffer_length;
    context.rx.start = rx_buffer;
    context.rx.head = context.rx.start;
    context.rx.tail = context.rx.start;
    context.rx.end = rx_buffer + rx_buffer_length;
    context.rx.bytes_available = 0;
    context.tx.bytes_available = -1;
    context.tx.next_packet_id = 1;
    context.tx.active_data_packet_id = 0;
    context.rx.previous_data_packet_id = 0;

    uart_stream_internal_set_irq_enabled(true);

    return true;
}


/*************************************************************************************************/
bool is_synchronized()
{
    bool synced;

    CORE_ATOMIC_SECTION(
        synced = !context.sync_requested && context.tx.bytes_available >= 0;
    );

    return synced;
}

/*************************************************************************************************/
int32_t get_rx_bytes_available()
{
    int32_t bytes_available;

    CORE_ATOMIC_SECTION(
        bytes_available = (!context.sync_requested && context.tx.bytes_available >= 0) ? context.rx.bytes_available - context.rx.dma_bytes_remaining : -1;
    );

    return bytes_available;
}


/*************************************************************************************************/
int32_t get_tx_bytes_available()
{
    int32_t bytes_available;

    CORE_ATOMIC_SECTION(
        bytes_available =
        context.sync_requested ? -1 :
        ((context.tx.active_data_packet_id > 0) || (context.tx.dma_bytes_remaining > 0)) ? 0 :
        context.tx.bytes_available;
    );

    return bytes_available;
}

/*************************************************************************************************/
bool synchronize()
{
    bool sync_requested;

    CORE_ATOMIC_SECTION(
        sync_requested = context.sync_requested;
        context.sync_requested = false;

        if(sync_requested || context.tx.bytes_available == -1)
        {
            context.tx.active_data_packet_id = 0;
        }
    );

    if(sync_requested)
    {
        write_data_packet_header(ACKNOWLEDGE_SYNCHRONIZATION);
        return false;
    }
    else if(!is_synchronized())
    {
        write_data_packet_header(REQUEST_SYNCHRONIZATION);
        return false;
    }
    else
    {
        return true;
    }
}

/*************************************************************************************************/
int32_t write(const void *data, int32_t max_length, bool block)
{
    if(!synchronize())
    {
        return -1;
    }

    const int32_t max_tx_bytes = get_tx_bytes_available();
    int32_t tx_length = std::min(max_tx_bytes, (int32_t)max_length);

    if(tx_length <= 0)
    {
        return tx_length;
    }

    CORE_ATOMIC_SECTION(
        context.tx.dma_bytes_remaining = tx_length;
        context.tx.dma_ptr = (const uint8_t*)data;
    );

    uint8_t tx_packet_id = write_data_packet_header(tx_length, false);
    CORE_ATOMIC_SECTION(
        context.tx.active_data_packet_id = tx_packet_id;
    );


    start_next_tx_dma_transfer();

    if(block)
    {
        wait_for_tx_dma_to_complete();
    }

    return tx_length;
}

/*************************************************************************************************/
bool write_cmd(uint8_t code, const uint8_t payload[COMMAND_PAYLOAD_LENGTH])
{
    if(!synchronize())
    {
        return false;
    }

    const PacketHeader header(-1, code, payload);
    wait_for_tx_dma_to_complete();
    uart_stream_internal_write_data((const uint8_t*)&header, sizeof(header));

    return true;
}

/*************************************************************************************************/
bool read_cmd(uint8_t* code, uint8_t payload[COMMAND_PAYLOAD_LENGTH])
{
    if(!synchronize())
    {
        return false;
    }
    else if(!context.rx.cmd_available)
    {
        return false;
    }
    else
    {
        context.rx.cmd_available = false;
        *code = context.rx.cmd_code;
        if(payload != nullptr)
        {
            memcpy(payload, context.rx.cmd_payload, COMMAND_PAYLOAD_LENGTH);
        }

        return true;
    }
}

/*************************************************************************************************/
int32_t read(void *data, int32_t max_length)
{
    if(!synchronize())
    {
        return -1;
    }

    const int32_t rx_bytes_available = get_rx_bytes_available();
    max_length = (max_length >= 0) ? max_length : rx_bytes_available;
    const int32_t rx_length = std::min(max_length, rx_bytes_available);

    if(rx_length > 0)
    {
        int32_t rx_remaining = rx_length;
        uint8_t* dst = (uint8_t*)data;
        const uintptr_t length_to_end = (uintptr_t)(context.rx.end - context.rx.head);
        const int32_t chunk_length = std::min(rx_remaining, (int32_t)length_to_end);
        std::memcpy(dst, context.rx.head, chunk_length);
        context.rx.head += chunk_length;

        if(context.rx.head >= context.rx.end)
        {
            context.rx.head = context.rx.start;
            dst += chunk_length;
            rx_remaining -= chunk_length;
            std::memcpy(dst, context.rx.head, rx_remaining);
            context.rx.head += rx_remaining;
        }

        CORE_ATOMIC_SECTION(
            assert( context.rx.bytes_available >= rx_length);
            context.rx.bytes_available -= rx_length;
        );

        write_data_packet_header(0);
    }

    return rx_length;
}

/*************************************************************************************************/
int32_t read_direct(uint8_t** data_ptr, int32_t max_length)
{
    if(!synchronize())
    {
        return -1;
    }

    const int32_t rx_bytes_available = get_rx_bytes_available();
    max_length = (max_length >= 0) ? max_length : rx_bytes_available;

    const uintptr_t length_to_end = (uintptr_t)(context.rx.end - context.rx.head);
    const int32_t rx_length = std::min(std::min(max_length, rx_bytes_available), (int32_t)length_to_end);

    if(rx_length > 0)
    {
        *data_ptr = context.rx.head;
        context.rx.head += rx_length;

        if(context.rx.head >= context.rx.end)
        {
            context.rx.head = context.rx.start;
        }

        CORE_ATOMIC_SECTION(
            assert( context.rx.bytes_available >= rx_length);
            context.rx.bytes_available -= rx_length;
        );

        write_data_packet_header(0);
    }

    return rx_length;
}

/*************************************************************************************************/
int32_t flush(int32_t max_length)
{
    if(!synchronize())
    {
        return -1;
    }

    const int32_t rx_bytes_available = get_rx_bytes_available();
    max_length = (max_length >= 0) ? max_length : rx_bytes_available;
    int32_t rx_remaining = max_length;

    while(max_length > 0)
    {
        uint8_t* p;

        int32_t rx_length = read_direct(&p, max_length);
        if(rx_length == -1)
        {
            return -1;
        }
        max_length -= rx_length;
    }

    return max_length;
}

/*************************************************************************************************/
extern "C" void uart_stream_internal_rx_irq_callback()
{
    int32_t c = uart_stream_internal_read_char();
    if(c == -1)
    {
        return;
    }

    context.rx.header_buffer[sizeof(PacketHeader)-1] = c;

    PacketHeader& header = *(PacketHeader*)context.rx.header_buffer;

    if(!(header.id != 0 &&
         std::memcmp(header.delimiter2, PACKET_DELIMITER2, sizeof(PACKET_DELIMITER2)) == 0 &&
         std::memcmp(header.delimiter1, PACKET_DELIMITER1, sizeof(PACKET_DELIMITER1)) == 0))
    {
        std::memmove(context.rx.header_buffer, &context.rx.header_buffer[1], sizeof(PacketHeader)-1);
        return;
    }

    if(header.id < 0)
    {
        context.rx.cmd_available = true;
        context.rx.cmd_code = header.cmd.code;
        memcpy(context.rx.cmd_payload, header.cmd.payload, COMMAND_PAYLOAD_LENGTH);
        memset(context.rx.header_buffer, 0, sizeof(PacketHeader));
        return;
    }
    else if(!(header.data.rx_buffer_available >= 0 && header.data.packet_length >= -1))
    {
        return;
    }

    CORE_ATOMIC_SECTION(
        if(header.data.ack_id == context.tx.active_data_packet_id)
        {
            context.tx.active_data_packet_id = 0;
        }

        if(header.data.packet_length == REQUEST_SYNCHRONIZATION)
        {
            header.data.packet_length = 0;
            context.sync_requested = true;
            context.tx.active_data_packet_id = 0;
        }

        if(header.data.packet_length > 0)
        {
            context.rx.previous_data_packet_id = header.id;
        }

        context.rx.dma_bytes_remaining = header.data.packet_length;
        context.rx.bytes_available += header.data.packet_length;
        context.tx.bytes_available = header.data.rx_buffer_available;
        assert(get_rx_bytes_available() <= context.rx.buffer_length);
    );

    const bool start_rx_dma = header.data.packet_length > 0;
    memset(context.rx.header_buffer, 0, sizeof(PacketHeader));

    if(start_rx_dma)
    {
        start_next_rx_dma_transfer();
    }
}

/*************************************************************************************************/
static bool start_next_rx_dma_transfer(unsigned int channel,unsigned int sequenceNo, void *userParam)
{
    if(context.rx.dma_chunk_length > 0)
    {
        context.rx.dma_bytes_remaining -= context.rx.dma_chunk_length;
        assert(context.rx.dma_bytes_remaining >= 0);
        context.rx.tail += context.rx.dma_chunk_length;
        if(context.rx.tail >= context.rx.end)
        {
            context.rx.tail = context.rx.start;
        }
        context.rx.dma_chunk_length = 0;
    }

    uintptr_t length_to_end = (context.rx.end - context.rx.tail);
    uint32_t chunk_length = std::min(std::min((uint32_t)length_to_end, (uint32_t)context.rx.dma_bytes_remaining), (uint32_t)MAX_DMA_LENGTH);
    if(chunk_length == 0)
    {
        if(context.tx.dma_bytes_remaining == 0)
        {
            write_data_packet_header(0);
        }
        else
        {
            context.tx.send_ack_after_transfer = true;
        }

        uart_stream_internal_set_irq_enabled(true);
        return true;
    }

    LDMA_Descriptor_t desc = LDMA_DESCRIPTOR_SINGLE_P2M_BYTE(
        context.dma_config.rx_address,
        context.rx.tail,
        chunk_length
    );
    desc.xfer.doneIfs = 1;
    context.rx.dma_chunk_length = chunk_length;

    uart_stream_internal_set_irq_enabled(false);
    DMADRV_LdmaStartTransfer(context.rx.dma_channel, &context.dma_config.rx_cfg, &desc, start_next_rx_dma_transfer, nullptr);

    return true;
}

/*************************************************************************************************/
static bool start_next_tx_dma_transfer(unsigned int unused,unsigned int unused2, void *unused3)
{
    if(context.tx.dma_chunk_length > 0)
    {
        context.tx.dma_bytes_remaining -= context.tx.dma_chunk_length;
        assert(context.tx.dma_bytes_remaining >= 0);
        context.tx.dma_ptr += context.tx.dma_chunk_length;
        context.tx.dma_chunk_length = 0;
        if(context.tx.dma_bytes_remaining == 0)
        {
            context.tx.dma_ptr = nullptr;
            if(context.tx.send_ack_after_transfer)
            {
                write_data_packet_header(0);
            }
            return true;
        }
    }

    const uint32_t chunk_length = std::min((uint32_t)MAX_DMA_LENGTH, (uint32_t)context.tx.dma_bytes_remaining);
    LDMA_Descriptor_t desc = LDMA_DESCRIPTOR_SINGLE_M2P_BYTE(
        context.tx.dma_ptr,
        context.dma_config.tx_address,
        chunk_length
    );
    desc.xfer.doneIfs = 1;
    context.tx.dma_chunk_length = chunk_length;

    DMADRV_LdmaStartTransfer(context.tx.dma_channel, &context.dma_config.tx_cfg, &desc, start_next_tx_dma_transfer, nullptr);

    return true;
}


/*************************************************************************************************/
static int8_t write_data_packet_header(int32_t tx_length, bool block)
{
    volatile int32_t unused_rx_buffer_length;
    uint8_t packet_id;
    uint8_t ack_id;

    CORE_ATOMIC_SECTION(
        unused_rx_buffer_length = context.rx.buffer_length - context.rx.bytes_available;
        packet_id = context.tx.next_packet_id++;
        ack_id = context.rx.previous_data_packet_id;
        context.tx.next_packet_id = std::max(context.tx.next_packet_id, (int8_t)1);
        context.tx.send_ack_after_transfer = false;
    );

    const PacketHeader header(packet_id, ack_id, tx_length, unused_rx_buffer_length);
    if(block)
    {
        wait_for_tx_dma_to_complete();
    }

    uart_stream_internal_write_data((const uint8_t*)&header, sizeof(header));

    return packet_id;
}

/*************************************************************************************************/
static void wait_for_tx_dma_to_complete()
{
    while(context.tx.dma_bytes_remaining > 0)
    {
        EMU_EnterEM1();
    }
}

} // namespace uart_stream