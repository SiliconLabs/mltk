/***************************************************************************//**
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/stat.h>
#include <sys/times.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/unistd.h>
#include <string.h>



#include "sl_stdio.h"
#include "microsecond_timer.h"


#if __has_include ("sl_iostream_eusart_vcom_config.h")
#include "em_eusart.h"
#include "sl_iostream_eusart_vcom_config.h"
#define USART_TX(c) EUSART_Tx(SL_IOSTREAM_EUSART_VCOM_PERIPHERAL, c)

#elif __has_include("sl_iostream_usart_vcom_config.h")
#include "em_usart.h"
#include "sl_iostream_usart_vcom_config.h"
#define USART_TX(c) USART_Tx(SL_IOSTREAM_USART_VCOM_PERIPHERAL, c)

#else 
#warning Failed to determine USART for _asset_write
#endif


extern void* heap_malloc(uint32_t size); 
extern void heap_free(void* ptr);



/*************************************************************************************************/
void _asset_write(const char *s)
{
#ifdef USART_TX
  while(*s != 0)
  {
    USART_TX(*s++);
  }
#endif
}

/*************************************************************************************************/
void _abort(void)
{
    assert(!"Program aborted");
    for(;;);
}

/*************************************************************************************************/
struct tm* localtime(const time_t *_timer)
{
    static struct tm tm_time;


    memset(&tm_time, 0, sizeof(tm_time));

    return &tm_time;
}

/*************************************************************************************************/
int _gettimeofday(struct timeval *tv, struct timezone *tz)
{
  const uint64_t us = microsecond_timer_get_timestamp();
  const div_t qr = div(us, 1000000); // convert to seconds
  tv->tv_sec = qr.quot;
  tv->tv_usec = qr.rem; 
  return 0;  // return non-zero for error
}

/*************************************************************************************************/
void __assert_func( const char * file, int line, const char * func, const char * failedexpr )
{
    char buffer[512];
    /* Assertion failed!
     *
     * To find out where this assert was triggered, either look up the call stack,
     * or inspect the file, line and function parameters
     */
    snprintf(buffer, sizeof(buffer)-1, "\n\n*** Assert Failed: %s:%d %s %s\n\n", file, line, func, failedexpr);
    buffer[sizeof(buffer)-1] = 0;
    _asset_write(buffer);

     __asm( "bkpt" );

    for(;;);
}


/*************************************************************************************************/
void* __wrap_malloc(uint32_t size)
{
  return heap_malloc(size);
}

/*************************************************************************************************/
void* __wrap__malloc_r(void *v, uint32_t size)
{
  return heap_malloc(size);
}

/*************************************************************************************************/
void* __wrap_calloc(uint32_t count, uint32_t size)
{
  return heap_malloc(count*size);
}

/*************************************************************************************************/
void* __wrap__calloc_r(void* x, uint32_t count, uint32_t size)
{
  return heap_malloc(count*size);
}

/*************************************************************************************************/
void __wrap_free(void* p)
{
  heap_free(p);
}

/*************************************************************************************************/
void __wrap__free_r(void* x, void* p)
{
  heap_free(p);
}