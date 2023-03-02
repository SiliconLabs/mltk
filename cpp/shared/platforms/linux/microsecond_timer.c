#include <sys/time.h>
#include <stdlib.h>
#include <time.h>

#include "microsecond_timer.h"



uint32_t microsecond_timer_get_timestamp()
{
    static uint32_t base_time = 0;
    struct timeval te;

    gettimeofday(&te, NULL); // get current time

    const uint32_t t = (uint32_t)(te.tv_sec*1000000ULL + te.tv_usec);
    if(base_time == 0)
    {
        base_time = t;
    }
    return t - base_time;
}
