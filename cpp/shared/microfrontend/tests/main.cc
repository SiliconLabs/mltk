#include <stdarg.h>
#include <stdio.h>

#include "sl_system_init.h"
#include "gtest/gtest.h"
#include "microfrontend/lib/frontend_util.h"


extern int _host_argc;
extern char** _host_argv;

extern "C" int main(int argc, char **argv) 
{
    sl_system_init();

#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
    ::testing::InitGoogleTest(&_host_argc, _host_argv);
#else 
    ::testing::InitGoogleTest();
#endif
    return RUN_ALL_TESTS();
}
