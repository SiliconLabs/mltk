
#include "microsecond_timer.h"

#if defined(_WIN32)
    #include <windows.h>

#elif defined(__unix__)
    #include <sys/time.h>
    #include <stdlib.h>
    #include <time.h>

#elif defined(__arm__)
    #include "sl_sleeptimer.h"
    #include "../src/sli_sleeptimer_hal.h"

    #if SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_RTCC
        #define MICROSECOND_TIMER_CNT RTCC->CNT

    #elif SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_BURTC
        #define MICROSECOND_TIMER_CNT BURTC->CNT

    #elif SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_RTC
        #define MICROSECOND_TIMER_CNT RTC->CNT

    #elif SL_SLEEPTIMER_PERIPHERAL == SL_SLEEPTIMER_PERIPHERAL_SYSRTC
        #define MICROSECOND_TIMER_CNT SYSRTC0->CNT

    #else 
        #error ARM platform not supported
    #endif

#else
    #error Platform not supported   
#endif


#if defined(_WIN32)
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


#elif defined(__unix__)
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

#elif defined(__arm__)
uint32_t microsecond_timer_get_timestamp()
{
    static uint32_t timer_scaler = 0;
    if(timer_scaler == 0)
    {
        const uint32_t timer_freq = sleeptimer_hal_get_timer_frequency();
        timer_scaler = 1000000 / timer_freq;
    }

    return MICROSECOND_TIMER_CNT * timer_scaler;
}
#endif 
