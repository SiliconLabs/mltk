#include <sys/time.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <errno.h>  

#include "sl_sleeptimer.h"


extern uint32_t periodic_wakeup_interval_ms;


uint32_t sl_sleeptimer_get_tick_count(void)
{
    static uint32_t base_time = 0;
    struct timeval te;

    gettimeofday(&te, NULL); // get current time

    const uint32_t t = te.tv_sec*1000ULL + te.tv_usec / 1000UL;
    if(base_time == 0)
    {
        base_time = t;
    }
    return t - base_time;
}


void sl_sleeptimer_delay_millisecond(uint16_t time_ms)
{
    struct timespec ts;
    int res;

    if (time_ms < 0)
    {
        errno = EINVAL;
        return;
    }

    ts.tv_sec = time_ms / 1000;
    ts.tv_nsec = (time_ms % 1000) * 1000000;

    do {
        res = nanosleep(&ts, &ts);
    } while (res && errno == EINTR);
}


uint32_t sl_sleeptimer_tick_to_ms(uint32_t tick)
{
    return tick;
}


sl_status_t sl_sleeptimer_start_periodic_timer_ms(sl_sleeptimer_timer_handle_t *handle,
                                                  uint32_t timeout_ms,
                                                  sl_sleeptimer_timer_callback_t callback,
                                                  void *callback_data,
                                                  uint8_t priority,
                                                  uint16_t option_flags)
{
    periodic_wakeup_interval_ms = timeout_ms;
    return SL_STATUS_OK;
}
