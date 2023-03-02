#include <stdint.h>
#include <stdio.h>

#include "sl_system_init.h"
#include "sl_sleeptimer.h"


uint32_t periodic_wakeup_interval_ms = 0;
static uint32_t periodic_wakeup_timestamp = 0;
static uint32_t periodic_wakeup_overflow_count = 0;
static int printed_latency_msg = 0;




void sl_system_init(void)
{

}

void sl_system_process_action(void)
{
    if(periodic_wakeup_interval_ms == 0)
    {
        return;
    }

    const uint32_t now = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
    const uint32_t elapsed = now - periodic_wakeup_timestamp;

    if(elapsed <= periodic_wakeup_interval_ms)
    {
        periodic_wakeup_overflow_count = 0;
        const int32_t remaining = (periodic_wakeup_interval_ms - elapsed);
        if(remaining > 0)
        {
            sl_sleeptimer_delay_millisecond(remaining);
        }
    }
    else if(!printed_latency_msg)
    {
        periodic_wakeup_overflow_count++;
        if(periodic_wakeup_overflow_count >= 5)
        {
            printed_latency_msg = 1;
            printf("\n*** Simulated latency is %dms, but app loop took %dms", periodic_wakeup_interval_ms, elapsed);
            printf("This likely means the model is taking too long to execute on your PC\n");
        }
    }

    periodic_wakeup_timestamp = sl_sleeptimer_tick_to_ms(sl_sleeptimer_get_tick_count());
}
