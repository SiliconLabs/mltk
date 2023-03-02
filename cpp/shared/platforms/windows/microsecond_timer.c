#include <windows.h>


#include "microsecond_timer.h"



uint32_t microsecond_timer_get_timestamp()
{
    static LARGE_INTEGER frequency = {0ULL};
    static uint32_t base_time = 0;
    LARGE_INTEGER current_time;
    if(frequency.QuadPart == 0ULL)
    {
        QueryPerformanceFrequency(&frequency);
    }

    QueryPerformanceCounter(&current_time);
    const uint32_t t = (uint32_t)((current_time.QuadPart * 1000000) / frequency.QuadPart);
    if(base_time == 0)
    {
        base_time = t;
    }
    return t - base_time;
}
