
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "stacktrace/stacktrace.h"




/*************************************************************************************************/
void __wrap___assert_fail (const char *__assertion, const char *__file,
			   unsigned int __line, const char *__function)
{
    /* Assertion failed!
     *
     * To find out where this assert was triggered, either look up the call stack,
     * or inspect the file, line and function parameters
     */
    printf("Assertion failed: %s (%d): %s %s\n\n", __file, __line, __function, __assertion);
    fflush(stdout);

    stacktrace_dump();

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

    exit(-1);
}


/*************************************************************************************************/
void __wrap___assert_perror_fail (int __errnum, const char *__file,
				  unsigned int __line, const char *__function)
{
    /* Assertion failed!
     *
     * To find out where this assert was triggered, either look up the call stack,
     * or inspect the file, line and function parameters
     */
    printf("Assertion failed: %s (%d): %s %d\n\n", __file, __line, __function, __errnum);
    fflush(stdout);

    stacktrace_dump();

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