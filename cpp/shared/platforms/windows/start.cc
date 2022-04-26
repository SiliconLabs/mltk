#include <assert.h>
#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include "stacktrace/stacktrace.h"
#include "cpputils/heap.hpp"


int _host_argc;
char** _host_argv;


extern "C" int __real_main(void);
//extern "C" void* __real_malloc(uint32_t);

extern "C" int __wrap_main(int argc, char **argv)
{
    stacktrace_init();

    // https://www.cplusplus.com/reference/cstdio/setvbuf/
    // flush on every newline
    setvbuf(stdout, NULL, _IOLBF, 0); 

    const uint32_t heap_size = 64*1024*1024;
    void* heap_base = malloc(heap_size);
    heap_set_buffer(heap_base, heap_size);
    _host_argc = argc;
    _host_argv = argv;

    return __real_main();
}
