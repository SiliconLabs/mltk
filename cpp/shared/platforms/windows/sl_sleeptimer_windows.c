#include <windows.h>
#include <pthread.h>

#include "sl_sleeptimer.h"


extern uint32_t periodic_wakeup_interval_ms;


uint32_t sl_sleeptimer_get_tick_count(void)
{
    static uint32_t base_time = 0;
    SYSTEMTIME time;
    GetSystemTime(&time);
    const uint32_t t = (uint32_t)((time.wHour * 360000) + (time.wMinute * 60000) + (time.wSecond * 1000) + time.wMilliseconds);
    if(base_time == 0)
    {
        base_time = t;
    }
    return t - base_time;
}


void sl_sleeptimer_delay_millisecond(uint16_t time_ms)
{
    const struct timespec delay = {0, (long)(time_ms * 1000*1000)};
    pthread_delay_np(&delay);
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
