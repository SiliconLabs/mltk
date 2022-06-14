#include <stdarg.h>
#include <stdio.h>


#include "gtest/gtest.h"
#include "microfrontend/lib/frontend_util.h"




extern "C" int main(int argc, char **argv) 
{
#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
    if(argc < 0 || argc > 50) { // if a bogus argc was passed in, then just clear it
        argc = 0;
        argv = nullptr;
    }
    ::testing::InitGoogleTest(&argc, argv);
#else 
    ::testing::InitGoogleTest();
#endif
    return RUN_ALL_TESTS();
}
