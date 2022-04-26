#pragma once 


#include "tflite_model_parameters/tflite_model_parameters.hpp"


namespace mltk
{

class TfliteMicroModel;

/**
 * @brief Model Details
 * 
 * Details about a loaded model
 */
struct TfliteMicroModelDetails
{
public:

    /**
     * The amount of working RAM required by the model
     * This is the minimum amount of memory required by 
     * the tflite-micro tensor "arena"
     */
    unsigned runtime_memory_size() const;

    /**
     * Model name found in model TfliteModelParameters
     */
    const char* name() const;

    /**
     * Name of accelerator used to optimize kernels
     */
    const char* accelerator() const;

    /**
     * Model version found in model TfliteModelParameters
     */
    unsigned version() const;

    /**
     * Model description found in model TfliteModelParameters
     */
    const char* description() const;

    /**
     * List of class names this model detects TfliteModelParameters
     */
    const StringList& classes() const;

    /**
     * Model date found in model TfliteModelParameters
     */
    const char* date() const;

    /**
     * Model unique hash found in model Metadata
     * The hash is calculated from the model's binary data and parameters.
     * It may be used to compare model flatbuffers for equality.
     */
    const char* hash() const;


protected:
    const char* _name = nullptr;
    const char* _date = nullptr;
    unsigned _version = 0;
    const char* _description = nullptr;
    const char* _hash = nullptr;
    const char* _accelerator = nullptr;
    TfliteModelParameters::StringList _classes;
    unsigned _runtime_memory_size = 0;

    bool load_parameters(const TfliteModelParameters *params);
    void unload();

    friend TfliteMicroModel;
};




} // namespace mltk
