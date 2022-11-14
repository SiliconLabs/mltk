#pragma once 

#include <cstdint>
#include <string>

#include "logging/logging.hpp"


#ifndef VERBOSE
#define VERBOSE false
#define VERBOSE_PROVIDED false
#else 
#define VERBOSE_PROVIDED true
#endif

struct CliOpts
{
    bool verbose = VERBOSE;
    const uint8_t* model_flatbuffer = nullptr;
    uint32_t model_flatbuffer_len = 0;
    bool verbose_provided = VERBOSE_PROVIDED;
    bool model_flatbuffer_provided = false;


    ~CliOpts();
};

extern CliOpts cli_opts;

void parse_cli_opts();

