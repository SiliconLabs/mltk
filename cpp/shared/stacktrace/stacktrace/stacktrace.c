
// https://gist.github.com/jvranish/4441299
// https://spin.atomicobject.com/2013/01/13/exceptions-stack-traces-c/


#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/stat.h>   // stat


#ifdef _WIN32
#include <windows.h>
#include <imagehlp.h>
#elif defined(__unix__)
#include <link.h>
#include <dlfcn.h>
#include <err.h>
#include <execinfo.h>
#include <unistd.h>
#else
#error OS not supported
#endif


#include "stacktrace/stacktrace.h"



#if INTPTR_MAX == INT64_MAX
#define IMAGE_FILE_MACHINE IMAGE_FILE_MACHINE_AMD64
#else
#define IMAGE_FILE_MACHINE IMAGE_FILE_MACHINE_I386
#endif

#ifdef _WIN32
#define ADDR2LINE_NAME "addr2line.exe"
#define ADDR2LINE_ARGS " -sipfC -e %s %p 2>&1"
#elif defined(__unix__)
#define ADDR2LINE_NAME "addr2line"
#define ADDR2LINE_ARGS " -s -i -f -C -p -e %s -a %s 2>&1"
#elif defined(__APPLE__)
#define ADD2LINE_NAME "atos"
#define ADDR2LINE_ARGS " -o %s %s 2>&1"
#endif // _WIN32


static int os_init();
static void os_print_stacktrace();

static uint8_t stacktrace_initialized = 0;
static uint8_t addr2line_invalid = 0;
static char stacktrace_program_path[2048] = {0};
static char addr2line_path[4096] = {0};


/*************************************************************************************************/
void stacktrace_init()
{
    if(!stacktrace_initialized)
    {
        stacktrace_initialized = os_init();
    }
}
/*************************************************************************************************/
void stacktrace_dump()
{
    if(stacktrace_initialized)
    {
        os_print_stacktrace();
    }
}





/*************************************************************************************************
 * Resolve symbol name and source location given the path to the executable
   and an address
 */
static int addr2line(const char* addr, const char* program_path)
{
    static char cmd_buffer[4096] = {0};
    static char out_buffer[1024] = {0};

    if(addr2line_invalid)
    {
        return -1;
    }

    strcpy(cmd_buffer, addr2line_path);
    const int cmd_buffer_offset = strlen(cmd_buffer);
    snprintf(&cmd_buffer[cmd_buffer_offset], sizeof(cmd_buffer)- cmd_buffer_offset, " " ADDR2LINE_ARGS, program_path, addr);

    //fputs(cmd_buffer, stdout);
    //fflush(stdout);

    FILE *fp = popen(cmd_buffer, "r");
    if (fp == NULL)
    {
        printf("Stack tracing command failed: %s\n", cmd_buffer);
        addr2line_invalid = 1;
        return -1;
    }

    char* unused = fgets(out_buffer, sizeof(out_buffer), fp); (void)unused;
    pclose(fp);


    if((strstr(out_buffer, "is not recognized as an internal or external command") != NULL) ||
       (strstr(out_buffer, "No such file") != NULL) ||
       (strstr(out_buffer, "cannot find the path specified") != NULL))
    {
        printf("Stack tracing command failed: %s, err: %s\n", cmd_buffer, out_buffer);
        if(strstr(out_buffer, "is not recognized") != NULL)
        {
            printf("HINT: Add the compiler's \"bin\" directory to the environment \"PATH\" to view the full stacktrace of this exception\n");
        }

        addr2line_invalid = 1;
        return -1;
    }
    else if(strstr(out_buffer, "?? ??:0") != NULL)
    {
        // printf("%s\n", cmd_buffer);
        return -1;
    }

    fputs(out_buffer, stdout);

    return 0;
}



#ifdef _WIN32


/*************************************************************************************************/
static void os_print_stacktrace()
{
    // Trigger a breakpoint exception which will cause the stacktrace to print
    __asm__("int $3");
}


/*************************************************************************************************/
static void windows_print_stacktrace(CONTEXT *context)
{
    SymInitialize(GetCurrentProcess(), 0, true);

    STACKFRAME frame = { 0 };

    /* setup initial stack frame */
#if INTPTR_MAX == INT64_MAX
    frame.AddrPC.Offset         = context->Rip;
    frame.AddrStack.Offset      = context->Rsp;
    frame.AddrFrame.Offset      = context->Rbp;
#else
    frame.AddrPC.Offset         = context->Eip;
    frame.AddrStack.Offset      = context->Esp;
    frame.AddrFrame.Offset      = context->Ebp;
#endif
    frame.AddrPC.Mode           = AddrModeFlat;
    frame.AddrStack.Mode        = AddrModeFlat;
    frame.AddrFrame.Mode        = AddrModeFlat;

    printf("Stacktrace '%s':\n", stacktrace_program_path);

    while (StackWalk(IMAGE_FILE_MACHINE,
            GetCurrentProcess(),
            GetCurrentThread(),
            &frame,
            context,
            0,
            SymFunctionTableAccess,
            SymGetModuleBase,
            0 ) )
    {
        if(addr2line((void*)frame.AddrPC.Offset, stacktrace_program_path) < 0)
        {
            printf("  %p\n", (void*)frame.AddrPC.Offset);
        }
    }
    fflush(stdout);

    SymCleanup( GetCurrentProcess() );
}

/*************************************************************************************************/
static LONG WINAPI windows_exception_handler(EXCEPTION_POINTERS * ExceptionInfo)
{
    switch(ExceptionInfo->ExceptionRecord->ExceptionCode)
    {
    case EXCEPTION_ACCESS_VIOLATION:
        fputs("Error: EXCEPTION_ACCESS_VIOLATION\n", stdout);
        break;
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
        fputs("Error: EXCEPTION_ARRAY_BOUNDS_EXCEEDED\n", stdout);
        break;
    case EXCEPTION_BREAKPOINT:
        fputs("Error: EXCEPTION_BREAKPOINT\n", stdout);
        break;
    case EXCEPTION_DATATYPE_MISALIGNMENT:
        fputs("Error: EXCEPTION_DATATYPE_MISALIGNMENT\n", stdout);
        break;
    case EXCEPTION_FLT_DENORMAL_OPERAND:
        fputs("Error: EXCEPTION_FLT_DENORMAL_OPERAND\n", stdout);
        break;
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
        fputs("Error: EXCEPTION_FLT_DIVIDE_BY_ZERO\n", stdout);
        break;
    case EXCEPTION_FLT_INEXACT_RESULT:
        fputs("Error: EXCEPTION_FLT_INEXACT_RESULT\n", stdout);
        break;
    case EXCEPTION_FLT_INVALID_OPERATION:
        fputs("Error: EXCEPTION_FLT_INVALID_OPERATION\n", stdout);
        break;
    case EXCEPTION_FLT_OVERFLOW:
        fputs("Error: EXCEPTION_FLT_OVERFLOW\n", stdout);
        break;
    case EXCEPTION_FLT_STACK_CHECK:
        fputs("Error: EXCEPTION_FLT_STACK_CHECK\n", stdout);
        break;
    case EXCEPTION_FLT_UNDERFLOW:
        fputs("Error: EXCEPTION_FLT_UNDERFLOW\n", stdout);
        break;
    case EXCEPTION_ILLEGAL_INSTRUCTION:
        fputs("Error: EXCEPTION_ILLEGAL_INSTRUCTION\n", stdout);
        break;
    case EXCEPTION_IN_PAGE_ERROR:
        fputs("Error: EXCEPTION_IN_PAGE_ERROR\n", stdout);
        break;
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
        fputs("Error: EXCEPTION_INT_DIVIDE_BY_ZERO\n", stdout);
        break;
    case EXCEPTION_INT_OVERFLOW:
        fputs("Error: EXCEPTION_INT_OVERFLOW\n", stdout);
        break;
    case EXCEPTION_INVALID_DISPOSITION:
        fputs("Error: EXCEPTION_INVALID_DISPOSITION\n", stdout);
        break;
    case EXCEPTION_NONCONTINUABLE_EXCEPTION:
        fputs("Error: EXCEPTION_NONCONTINUABLE_EXCEPTION\n", stdout);
        break;
    case EXCEPTION_PRIV_INSTRUCTION:
        fputs("Error: EXCEPTION_PRIV_INSTRUCTION\n", stdout);
        break;
    case EXCEPTION_SINGLE_STEP:
        fputs("Error: EXCEPTION_SINGLE_STEP\n", stdout);
        break;
    case EXCEPTION_STACK_OVERFLOW:
        fputs("Error: EXCEPTION_STACK_OVERFLOW\n", stdout);
        break;
    default:
        fputs("Error: Unrecognized Exception\n", stdout);
        break;
    }
    fflush(stdout);
    /* If this is a stack overflow then we can't walk the stack, so just show
     where the error happened */
    if (EXCEPTION_STACK_OVERFLOW != ExceptionInfo->ExceptionRecord->ExceptionCode)
    {
        windows_print_stacktrace(ExceptionInfo->ContextRecord);
    }
    else
    {
#if INTPTR_MAX == INT64_MAX
        addr2line((void*)ExceptionInfo->ContextRecord->Rip, stacktrace_program_path);
#else
        addr2line((void*)ExceptionInfo->ContextRecord->Eip, stacktrace_program_path);
#endif
    }

    fflush(stdout);

    return EXCEPTION_EXECUTE_HANDLER;
}

/*************************************************************************************************/
static const char* rstrstr(const char* haystack, const char* needle)
{
  int needle_length = strlen(needle);
  const char* haystack_end = haystack + strlen(haystack) - needle_length;
  const char* p;
  size_t i;

  for(p = haystack_end; p >= haystack; --p)
  {
    for(i = 0; i < needle_length; ++i) {
      if(p[i] != needle[i])
        goto next;
    }
    return p;

    next:;
  }
  return NULL;
}

/*************************************************************************************************/
static int os_init()
{
    HMODULE hModule;
    struct stat buffer;   

    // Try to find the addr2line.exe executable
    strcpy(addr2line_path, ADDR2LINE_NAME);

    if(stat(addr2line_path, &buffer) != 0)
    {
        const char* USERPROFILE_DIR = getenv("USERPROFILE");
        if(USERPROFILE_DIR != NULL)
        {
            snprintf(addr2line_path, sizeof(addr2line_path), "%s\\.mltk\\tools\\toolchains\\gcc\\windows\\8.4.0\\mingw64\\bin\\addr2line.exe", USERPROFILE_DIR);
            if(stat(addr2line_path, &buffer) != 0)
            {
                // If we can't find it, then just revert to the default location
                strcpy(addr2line_path, ADDR2LINE_NAME);
            }
        }
    }


    if(GetModuleHandleEx(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            (LPCSTR)&stacktrace_init, &hModule))
    {
        if(GetModuleFileName(hModule, stacktrace_program_path, sizeof(stacktrace_program_path)))
        {
            SetUnhandledExceptionFilter(windows_exception_handler);
            //printf("path: %s\n", stacktrace_program_path);
            //fflush(stdout);
            return 1;
        }
    }

    return 0;
}



#else // defined(__unix__)


static void posix_signal_handler(int sig, siginfo_t *siginfo, void *context)
{
    (void)context;
    switch(sig)
    {
    case SIGSEGV:
        fputs("Caught SIGSEGV: Segmentation Fault\n", stderr);
        break;
    case SIGINT:
        // Don't print a stacktrace in this case
        goto exit;
        //fputs("Caught SIGINT: Interactive attention signal, (usually ctrl+c)\n", stderr);
        //break;
    case SIGFPE:
        switch(siginfo->si_code)
        {
        case FPE_INTDIV:
            fputs("Caught SIGFPE: (integer divide by zero)\n", stderr);
            break;
        case FPE_INTOVF:
            fputs("Caught SIGFPE: (integer overflow)\n", stderr);
            break;
        case FPE_FLTDIV:
            fputs("Caught SIGFPE: (floating-point divide by zero)\n", stderr);
            break;
        case FPE_FLTOVF:
            fputs("Caught SIGFPE: (floating-point overflow)\n", stderr);
            break;
        case FPE_FLTUND:
            fputs("Caught SIGFPE: (floating-point underflow)\n", stderr);
            break;
        case FPE_FLTRES:
            fputs("Caught SIGFPE: (floating-point inexact result)\n", stderr);
            break;
        case FPE_FLTINV:
            fputs("Caught SIGFPE: (floating-point invalid operation)\n", stderr);
            break;
        case FPE_FLTSUB:
            fputs("Caught SIGFPE: (subscript out of range)\n", stderr);
            break;
        default:
            fputs("Caught SIGFPE: Arithmetic Exception\n", stderr);
            break;
        }
        case SIGILL:
            switch(siginfo->si_code)
            {
            case ILL_ILLOPC:
                fputs("Caught SIGILL: (illegal opcode)\n", stderr);
                break;
            case ILL_ILLOPN:
                fputs("Caught SIGILL: (illegal operand)\n", stderr);
                break;
            case ILL_ILLADR:
                fputs("Caught SIGILL: (illegal addressing mode)\n", stderr);
                break;
            case ILL_ILLTRP:
                fputs("Caught SIGILL: (illegal trap)\n", stderr);
                break;
            case ILL_PRVOPC:
                fputs("Caught SIGILL: (privileged opcode)\n", stderr);
                break;
            case ILL_PRVREG:
                fputs("Caught SIGILL: (privileged register)\n", stderr);
                break;
            case ILL_COPROC:
                fputs("Caught SIGILL: (coprocessor error)\n", stderr);
                break;
            case ILL_BADSTK:
                fputs("Caught SIGILL: (internal stack error)\n", stderr);
                break;
            default:
                fputs("Caught SIGILL: Illegal Instruction\n", stderr);
                break;
            }
            break;
            case SIGTERM:
                // Don't print a stacktrace in this case
                goto exit;
                //fputs("Caught SIGTERM: a termination request was sent to the program\n", stderr);
                //break;
            case SIGABRT:
                fputs("Caught SIGABRT: usually caused by an abort() or assert()\n", stderr);
                break;
            default:
                break;
    }
    os_print_stacktrace();

exit:
    _Exit(1);
}


/*************************************************************************************************/
static int os_init()
{
    static uint8_t alternate_stack[SIGSTKSZ];

    strcpy(addr2line_path, ADDR2LINE_NAME);

    /* setup alternate stack */
    {
        stack_t ss = {};
        /* malloc is usually used here, I'm not 100% sure my static allocation
        is valid but it seems to work just fine. */
        ss.ss_sp = (void*)alternate_stack;
        ss.ss_size = SIGSTKSZ;
        ss.ss_flags = 0;

        if (sigaltstack(&ss, NULL) != 0) { err(1, "sigaltstack"); }
    }

    /* register our signal handlers */
    {
        struct sigaction sig_action = {};
        sig_action.sa_sigaction = posix_signal_handler;
        sigemptyset(&sig_action.sa_mask);

#ifdef __APPLE__
        /* for some reason we backtrace() doesn't work on osx
            when we use an alternate stack */
        sig_action.sa_flags = SA_SIGINFO;
#else
        sig_action.sa_flags = SA_SIGINFO | SA_ONSTACK;
#endif

        if (sigaction(SIGSEGV, &sig_action, NULL) != 0) { err(1, "sigaction"); }
        if (sigaction(SIGFPE,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
        if (sigaction(SIGINT,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
        if (sigaction(SIGILL,  &sig_action, NULL) != 0) { err(1, "sigaction"); }
        if (sigaction(SIGTERM, &sig_action, NULL) != 0) { err(1, "sigaction"); }
        if (sigaction(SIGABRT, &sig_action, NULL) != 0) { err(1, "sigaction"); }
        if (sigaction(SIGIOT, &sig_action, NULL) != 0) { err(1, "sigaction"); }
        if (sigaction(SIGTRAP, &sig_action, NULL) != 0) { err(1, "sigaction"); }
        if (sigaction(SIGBUS, &sig_action, NULL) != 0) { err(1, "sigaction"); }
    }

    int result = readlink("/proc/self/exe", stacktrace_program_path, sizeof(stacktrace_program_path)); 
    (void)result;

    return 1;
}

/*************************************************************************************************/
static void os_print_stacktrace()
{
#define MAX_STACK_FRAMES 64
    static void *stack_traces[MAX_STACK_FRAMES];


    int i, trace_size = 0;
    char **messages = (char **)NULL;

    trace_size = backtrace(stack_traces, MAX_STACK_FRAMES);
    messages = backtrace_symbols(stack_traces, trace_size);

    printf("Stacktrace:\n");

    /* skip the first couple stack frames (as they are this function and
     our handler) and also skip the last frame as it's (always?) junk. */
    // for (i = 3; i < (trace_size - 1); ++i)
    for (i = 0; i < trace_size; ++i) // we'll use this for now so you can see what's going on
    {
        char* start = strstr(messages[i], "(+");
        if(start != NULL)
        {
            start += 2;
            char* end = strstr(start, ")");
            if(end != NULL)
            {
                char path_buffer[1024];
                char addr_buffer[32];
                int path_len = (int)(start - messages[i]) - 2;
                int addr_len = (int)(end - start);

                strncpy(path_buffer, messages[i], path_len);
                path_buffer[path_len] = 0;

                strncpy(addr_buffer, start, addr_len);
                addr_buffer[addr_len] = 0;

                if (addr2line(addr_buffer, path_buffer) == 0)
                {
                    continue;
                }
            }
        }
        start = strstr(messages[i], "[");
        if(start != NULL)
        {
            start += 1;
            char* end = strstr(start, "]");
            if(end != NULL)
            {
                int addr_len = (int)(end - start);
                char addr_buffer[addr_len+1];

                strncpy(addr_buffer, start, addr_len);
                addr_buffer[addr_len] = 0;

                if (addr2line(addr_buffer, stacktrace_program_path) == 0)
                {
                    continue;
                }
            }
        }

        printf("  %s\n", messages[i]);
    }
    if (messages)
    {
        free(messages);
    }
}



#endif

