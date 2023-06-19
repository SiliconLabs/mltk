#pragma once


#if __has_include("platform_em_device.h")
#include "platform_em_device.h"
#endif

#if __has_include("simulator_em_device.h")
#include "simulator_em_device.h"
#endif

#ifndef SRAM_SIZE
#define SRAM_SIZE (64*1024*1024)
#endif