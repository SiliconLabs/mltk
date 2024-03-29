
#include <stdio.h>
#include <stdlib.h>

#include "stacktrace/stacktrace.h"




/*************************************************************************************************/
void __wrap___assert_func(const char * file, int line, const char * func, const char * failedexpr)
{
    /* Assertion failed!
     *
     * To find out where this assert was triggered, either look up the call stack,
     * or inspect the file, line and function parameters
     */
    printf("Assertion failed: %s (%d): %s %s\n\n", file, line, func, failedexpr);
    fflush(stdout);

    stacktrace_dump();

#if defined(DEBUG) && defined(_WIN32)
    __asm__("int $3");
#endif

    exit(-1);
}



/*************************************************************************************************/
void __wrap__assert(const char* message, const char* filename, unsigned line)
{
    /* Assertion failed!
     *
     * To find out where this assert was triggered, either look up the call stack,
     * or inspect the file, line and function parameters
     */
    printf("Assertion failed: %s (%d): %s\n\n", filename, line, message);
    fflush(stdout);

    stacktrace_dump();

#if defined(DEBUG) && defined(_WIN32)
    __asm__("int $3");
#endif
    exit(-1);
}


/*************************************************************************************************/
void __wrap__wassert(const wchar_t* message, const wchar_t* filename, unsigned line)
{
    /* Assertion failed!
     *
     * To find out where this assert was triggered, either look up the call stack,
     * or inspect the file, line and function parameters
     */
    printf("Assertion failed: %ls (%d): %ls\n\n", filename, line, message);
    fflush(stdout);

    stacktrace_dump();

#if defined(DEBUG) && defined(_WIN32)
    __asm__("int $3");
#endif
    exit(-1);
}

/*************************************************************************************************/
void __wrap_abort(void)
{
    printf("Program aborted!\n");
    fflush(stdout);

    stacktrace_dump();

#if defined(DEBUG) && defined(_WIN32)
    __asm__("int $3");
#endif
    exit(-1);
}