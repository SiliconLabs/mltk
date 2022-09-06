#include <string>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "cli_opts.hpp"
#include "mltk_tflite_micro_helper.hpp"


CliOpts cli_opts;

#include "cxxopts.hpp"

extern int _host_argc;
extern char** _host_argv;

/*************************************************************************************************/
void parse_cli_opts()
{
    cxxopts::Options options("Model Profiler", "Profile the given ML model and print its results");
    options.add_options()
        ("v,verbose", "Enable verbose logging")
        ("m,model", "Path to .tflite model file. Use built-in, default model if omitted", cxxopts::value<std::string>())
        ("h,help", "Print usage")
    ;

    try 
    {
        auto result = options.parse(_host_argc, _host_argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if(result.count("verbose"))
        {
            cli_opts.verbose = true;
            cli_opts.verbose_provided = true;
        }

        if(result.count("model"))
        {
            const char* path = result["model"].as<std::string>().c_str();
            auto fp = fopen(path,"rb");
            if(fp == nullptr)
            {
                MLTK_ERROR("Failed to open model file: %s", path);
                exit(-1);
            }

            fseek(fp, 0, SEEK_END); 
            auto file_size = ftell(fp); 
            fseek(fp, 0, SEEK_SET); 
            auto buffer = malloc(file_size); 
            auto result = fread(buffer, 1, file_size, fp);
            fclose(fp);
            if(result != file_size)
            {
                MLTK_ERROR("Failed to read model file: %s", path);
                exit(-1);
            }

            cli_opts.model_flatbuffer = (uint8_t*)buffer;
            cli_opts.model_flatbuffer_len = file_size;
            cli_opts.model_flatbuffer_provided = true;
        }
    } 
    catch(std::exception &e) 
    {
        std::cout << e.what() << std::endl;
        std::cout << options.help() << std::endl;
        exit(-1);
    }
}

/*************************************************************************************************/
CliOpts::~CliOpts()
{
    if(model_flatbuffer_provided)
    {
        free((void*)model_flatbuffer);
    }
}
