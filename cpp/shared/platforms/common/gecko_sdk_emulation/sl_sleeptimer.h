
#ifndef SL_SLEEPTIMER_H
#define SL_SLEEPTIMER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "sl_status.h"

/// @cond DO_NOT_INCLUDE_WITH_DOXYGEN
#define SL_SLEEPTIMER_NO_HIGH_PRECISION_HF_CLOCKS_REQUIRED_FLAG 0x01

#define SLEEPTIMER_ENUM(name) typedef uint8_t name; enum name##_enum
/// @endcond

/// Timestamp, wall clock time in seconds.
typedef uint32_t sl_sleeptimer_timestamp_t;



// Forward declaration
typedef struct sl_sleeptimer_timer_handle sl_sleeptimer_timer_handle_t;

/***************************************************************************//**
 * Typedef for the user supplied callback function which is called when
 * a timer expires.
 *
 * @param handle The timer handle.
 *
 * @param data An extra parameter for the user application.
 ******************************************************************************/
typedef void (*sl_sleeptimer_timer_callback_t)(sl_sleeptimer_timer_handle_t *handle, void *data);

/// @brief Timer structure for sleeptimer
struct sl_sleeptimer_timer_handle {
  uint8_t unused;
};


#ifdef __cplusplus
extern "C" {
#endif


uint32_t sl_sleeptimer_get_tick_count(void);


void sl_sleeptimer_delay_millisecond(uint16_t time_ms);


uint32_t sl_sleeptimer_tick_to_ms(uint32_t tick);


sl_status_t sl_sleeptimer_start_periodic_timer_ms(sl_sleeptimer_timer_handle_t *handle,
                                                  uint32_t timeout_ms,
                                                  sl_sleeptimer_timer_callback_t callback,
                                                  void *callback_data,
                                                  uint8_t priority,
                                                  uint16_t option_flags);

#ifdef __cplusplus
}
#endif


#endif // SL_SLEEPTIMER_H
