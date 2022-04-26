#include <signal.h>
#include <stdio.h>
#include "stacktrace/stacktrace.h"

int  divide_by_zero();
void cause_segfault();
void stack_overflow();
void infinite_loop();
void illegal_instruction();
void cause_calamity();



int main(int argc, char * argv[])
{
  (void)argc;
  printf("Begin stacktrace test\n");
  stacktrace_init();
  cause_calamity();

  puts("TEST FAILED");

  return 0;
}

void cause_calamity()
{
  /* uncomment one of the following error conditions to cause a calamity of 
   your choosing! */
  
  // (void)divide_by_zero();
  cause_segfault();
  // assert(false);
  // infinite_loop();
  // illegal_instruction();
  // stack_overflow();
}



int divide_by_zero()
{
  int a = 1;
  int b = 0; 
  return a / b;
}

void cause_segfault()
{
  int * p = (int*)0x12345678;
  *p = 0;
}

void stack_overflow();
void stack_overflow()
{
  int foo[1000];
  (void)foo;
  stack_overflow();
}

/* break out with ctrl+c to test SIGINT handling */
void infinite_loop()
{
  while(1) {};
}

void illegal_instruction()
{
  /* I couldn't find an easy way to cause this one, so I'm cheating */
  raise(SIGILL);
}