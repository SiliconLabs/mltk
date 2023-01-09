
#include <string>
#include <cstdio>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "cli_opts.hpp"
#include "mltk_tflite_micro_helper.hpp"


CliOpts cli_opts;

extern std::string dump_audio_dir;
extern std::string dump_raw_spectrograms_dir;
extern std::string dump_spectrograms_dir;


#ifndef __arm__
#include "cxxopts.hpp"

extern int _host_argc;
extern char** _host_argv;

/*************************************************************************************************/
void parse_cli_opts()
{
    cxxopts::Options options("Audio Classifier", "Classify streaming audio from a microphone using the given ML model");
    options.add_options()
        ("v,verbose", "Enable verbose logging")
        ("l,latency", "Number of ms to simulate per execution loo", cxxopts::value<uint32_t>())
        ("m,model", "Path to .tflite model file. Use built-in, default model if omitted", cxxopts::value<std::string>())
        ("w,window_duration", "Controls the smoothing. Longer durations (in milliseconds) will give a higher confidence that the results are correct, but may miss some commands", cxxopts::value<uint32_t>())
        ("c,count", "The minimum number of inference results to average when calculating the detection value", cxxopts::value<uint32_t>())
        ("t,threshold", "Minumum model output threshold for a class to be considered detected, 0-255. Higher values increase precision at the cost of recall", cxxopts::value<uint32_t>())
        ("s,suppression", "Amount of milliseconds to wait after a keyword is detected before detecting new keywords", cxxopts::value<uint32_t>())
        ("d,volume", "Increase/decrease microphone audio gain. 0 = no change, <0 decrease, >0 increase", cxxopts::value<int>())
        ("x,dump_audio", "Dump the raw audio samples to the given directory", cxxopts::value<std::string>())
        ("r,dump_raw_spectrograms", "Dump the raw (i.e. unquantized) generated spectorgrams to the given directory", cxxopts::value<std::string>())
        ("z,dump_spectrograms", "Dump the quantized generated spectorgrams to the given directory", cxxopts::value<std::string>())
        ("i,sensitivity", "Sensitivity of the activity indicator", cxxopts::value<float>())
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

        if(result.count("latency"))
        {
            cli_opts.latency_ms = result["latency"].as<uint32_t>();
            cli_opts.latency_ms_provided = true;

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
            cli_opts.model_flatbuffer_provided = true;
        }

        if(result.count("window_duration"))
        {
            cli_opts.average_window_duration_ms = result["window_duration"].as<uint32_t>();
            cli_opts.average_window_duration_ms_provided = true;
        }

        if(result.count("count"))
        {
            cli_opts.minimum_count = result["count"].as<uint32_t>();
            cli_opts.minimum_count_provided = true;
        }

        if(result.count("suppression"))
        {
            cli_opts.suppression_ms = result["suppression"].as<uint32_t>();
            cli_opts.suppression_ms_provided = true;
        }

        if(result.count("threshold"))
        {
            cli_opts.detection_threshold = result["threshold"].as<uint32_t>();
            cli_opts.detection_threshold_provided = true;
        }

        if(result.count("volume"))
        {
            cli_opts.volume_gain = result["volume"].as<int>();
            cli_opts.volume_gain_provided = true;
        }

        if(result.count("sensitivity"))
        {
            cli_opts.sensitivity = result["sensitivity"].as<float>();
            cli_opts.sensitivity_provided = true;
        }

        if(result.count("dump_audio"))
        {
            cli_opts.dump_audio = true;
            dump_audio_dir = result["dump_audio"].as<std::string>();
        }

        if(result.count("dump_raw_spectrograms"))
        {
            cli_opts.dump_raw_spectrograms = true;
            dump_raw_spectrograms_dir = result["dump_raw_spectrograms"].as<std::string>();
        }

        if(result.count("dump_spectrograms"))
        {
            cli_opts.dump_spectrograms = true;
            dump_spectrograms_dir = result["dump_spectrograms"].as<std::string>();
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

#endif // __arm__
