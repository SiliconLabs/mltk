#include <cstdio>


namespace logging 
{


#if defined(_WIN32) || defined(unix) || defined(__unix) || defined(__unix__)

void default_log_writer(const char *msg, int length, void *arg)
{
    fwrite(msg, 1, length, stdout);
    fflush(stdout);
}

#else 

#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
#endif

extern "C" int _write(int fd, const void * ptr, size_t len);

void default_log_writer(const char *msg, int length, void *arg)
{
    _write(STDOUT_FILENO, msg, length);
    fflush(stdout);
}

#endif

} // namespace logging

