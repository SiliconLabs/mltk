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


extern void* heap_malloc(uint32_t size); 
extern void heap_free(void* ptr);



int __dso_handle;


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

/*************************************************************************************************/
caddr_t _sbrk(int inc)
{
    assert(!"_sbrk");
    for(;;);
}

/*************************************************************************************************/
void _exit(int code)
{
    assert(!"Program exited");
    for(;;);
}

/*************************************************************************************************/
void _abort(void)
{
    assert(!"Program aborted");
    for(;;);
}

/**************************************************************************//**
 * @brief
 *  Close a file.
 *
 * @param[in] file
 *  File you want to close.
 *
 * @return
 *  Returns 0 when the file is closed.
 *****************************************************************************/
int _close(int file)
{
  (void) file;
  return 0;
}


/**************************************************************************//**
 * @brief
 *  Status of an open file.
 *
 * @param[in] file
 *  Check status for this file.
 *
 * @param[in] st
 *  Status information.
 *
 * @return
 *  Returns 0 when st_mode is set to character special.
 *****************************************************************************/
int _fstat(int file, struct stat *st)
{
  (void) file;
  st->st_mode = S_IFCHR;
  return 0;
}

/**************************************************************************//**
 * @brief Get process ID.
 *****************************************************************************/
int _getpid(void)
{
  return 1;
}

/**************************************************************************//**
 * @brief
 *  Query whether output stream is a terminal.
 *
 * @param[in] file
 *  Descriptor for the file.
 *
 * @return
 *  Returns 1 when query is done.
 *****************************************************************************/
int _isatty(int file)
{
  (void) file;
  return 1;
}

/**************************************************************************//**
 * @brief Send signal to process.
 * @param[in] pid Process id (not used).
 * @param[in] sig Signal to send (not used).
 *****************************************************************************/
int _kill(int pid, int sig)
{
  (void)pid;
  (void)sig;
  return -1;
}

/**************************************************************************//**
 * @brief
 *  Set position in a file.
 *
 * @param[in] file
 *  Descriptor for the file.
 *
 * @param[in] ptr
 *  Poiter to the argument offset.
 *
 * @param[in] dir
 *  Directory whence.
 *
 * @return
 *  Returns 0 when position is set.
 *****************************************************************************/
int _lseek(int file, int ptr, int dir)
{
  (void) file;
  (void) ptr;
  (void) dir;
  return 0;
}

/**************************************************************************//**
 * @brief
 *  Read from a file.
 *
 * @param[in] file
 *  Descriptor for the file you want to read from.
 *
 * @param[in] ptr
 *  Pointer to the chacaters that are beeing read.
 *
 * @param[in] len
 *  Number of characters to be read.
 *
 * @return
 *  Number of characters that have been read.
 *****************************************************************************/
int _read(int file, char *ptr, int len)
{
  int c, rxCount = 0;

  (void) file;

  while (len--) {
    if ((c = sl_getchar_std_wrapper()) != -1) {
      *ptr++ = c;
      rxCount++;
    } else {
      break;
    }
  }

  if (rxCount <= 0) {
    return -1;                        /* Error exit */
  }

  return rxCount;
}


/**************************************************************************//**
 * @brief
 *  Write to a file.
 *
 * @param[in] file
 *  Descriptor for the file you want to write to.
 *
 * @param[in] ptr
 *  Pointer to the text you want to write
 *
 * @param[in] len
 *  Number of characters to be written.
 *
 * @return
 *  Number of characters that have been written.
 *****************************************************************************/
int _write(int file, const char *ptr, int len)
{
  int txCount;

  (void) file;

  for (txCount = 0; txCount < len; txCount++) {
    sl_putchar_std_wrapper(*ptr++);
  }

  return len;
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
    int len = snprintf(buffer, sizeof(buffer), "\n\n*** Assert Failed: %s:%d %s %s\n\n", file, line, func, failedexpr);
    _write(0, buffer, len);

     __asm( "bkpt" );

    for(;;);
}