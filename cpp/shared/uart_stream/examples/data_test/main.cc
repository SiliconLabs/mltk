#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <algorithm>

#undef NDEBUG
#include <cassert>


#include "sl_system_init.h"
#include "uart_stream.hpp"






uint8_t uart_stream_rx_buffer[2048];
uint8_t tx_buffer[2048];
uint8_t rx_buffer[2048];
uint8_t tx_data_counter = 0;
uint8_t rx_data_counter = 0;
uint8_t tx_cmd_counter = 0;
uint8_t rx_cmd_counter = 0;
uint8_t rx_cmd_payload_counter = 0;
uint8_t tx_cmd_payload_counter = 0;



extern "C" int main(void)
{
  sl_system_init();

  printf("Uart stream data test starting\n");

  uart_stream::initialize(
    (uint8_t*)uart_stream_rx_buffer,
    sizeof(uart_stream_rx_buffer)
  );


  for(uint32_t loop_count = 0;; ++loop_count) 
  {
    int32_t bytes_available;
    if(!uart_stream::synchronize())
    {
      tx_data_counter = 0;
      rx_data_counter = 0;
      rx_cmd_counter = 0;
      tx_cmd_counter = 0;
      rx_cmd_payload_counter = 0;
      tx_cmd_payload_counter = 0;
      continue;
    }


    bytes_available = uart_stream::get_tx_bytes_available();
    if(bytes_available > 128)
    {
      uint8_t* p = tx_buffer;
      int32_t tx_len = std::min(bytes_available, (int32_t)sizeof(tx_buffer));
      for(int i = 0; i < tx_len; ++i)
      {
        *p++ = tx_data_counter++;
      }
      uart_stream::write(tx_buffer, tx_len, false);
    }

    bytes_available = uart_stream::get_rx_bytes_available();
    if(bytes_available >= 128)
    {
      uint8_t* p = rx_buffer;
      int32_t rx_len = std::min(bytes_available, (int32_t)sizeof(rx_buffer));
      uart_stream::read(rx_buffer, rx_len);
      for(int i = 0; i < rx_len; ++i)
      {
        if(*p != rx_data_counter)
        {
          assert(!"data error");
        }
        p++;
        rx_data_counter++;
      }
    }

    uint8_t rx_cmd_code;
    uint8_t rx_cmd_payload[uart_stream::COMMAND_PAYLOAD_LENGTH];
    if(uart_stream::read_cmd(&rx_cmd_code, rx_cmd_payload))
    {
      if(rx_cmd_code != rx_cmd_counter)
      {
        assert(!"cmd error");
      }
      rx_cmd_counter++;

      for(int i = 0; i < uart_stream::COMMAND_PAYLOAD_LENGTH; ++i)
      {
        if(rx_cmd_payload[i] != rx_cmd_payload_counter)
        {
          assert(!"cmd error");
        }
        rx_cmd_payload_counter ++;
      }
    }

    if(loop_count % 10000 == 0)
    {
      uint8_t payload[uart_stream::COMMAND_PAYLOAD_LENGTH];
      for(int i = 0; i < uart_stream::COMMAND_PAYLOAD_LENGTH; ++i)
      {
        payload[i] = tx_cmd_payload_counter++;
      }
      uart_stream::write_cmd(tx_cmd_counter++, payload);
    }

  }

  return 0;
}

