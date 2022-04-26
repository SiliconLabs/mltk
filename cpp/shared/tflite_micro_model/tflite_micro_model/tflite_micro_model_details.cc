#include "tflite_micro_model/tflite_micro_model_details.hpp"


namespace mltk
{

/*************************************************************************************************/
bool TfliteMicroModelDetails::load_parameters(const TfliteModelParameters*params)
{
    unload();

    params->get("name", _name);
    params->get("version", _version);
    params->get("classes", _classes);
    params->get("hash", _hash);
    params->get("date", _date);

    return true;
}

/*************************************************************************************************/
void TfliteMicroModelDetails::unload()
{
    _name = nullptr;
    _accelerator = nullptr;
    _date = nullptr;
    _version = 0;
    _description = nullptr;
    _hash = nullptr;
    _runtime_memory_size = 0;
    _classes.vector = nullptr;
}

/*************************************************************************************************/
unsigned TfliteMicroModelDetails::runtime_memory_size() const
{
    return _runtime_memory_size;
}


/*************************************************************************************************/
const char* TfliteMicroModelDetails::name() const
{
    return _name ? _name : "";
}

/*************************************************************************************************/
const char* TfliteMicroModelDetails::accelerator() const
{
    return _accelerator ? _accelerator : "none";
}

/*************************************************************************************************/
const char* TfliteMicroModelDetails::date() const
{
    return _date ? _date : "";
}

/*************************************************************************************************/
unsigned TfliteMicroModelDetails::version() const
{
    return _version;
}

/*************************************************************************************************/
const char* TfliteMicroModelDetails::description() const
{
    return _description ? _description : "";
}

/*************************************************************************************************/
const StringList& TfliteMicroModelDetails::classes() const
{
    return _classes;
}

/*************************************************************************************************/
const char* TfliteMicroModelDetails::hash() const
{
    return _hash ? _hash : "";
}


} // namespace mltk
