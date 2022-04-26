#pragma once 

#include <stdint.h>
#include <stdbool.h>



void jlink_stream_internal_init(uint32_t *trigger_address_ptr, uint32_t *trigger_value_ptr, uint32_t context_address);
void jlink_stream_set_interrupt_enabled(bool enabled);


