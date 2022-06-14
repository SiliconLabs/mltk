#pragma once
#include <cstdint>

#include "logging/logger.hpp"
#include "profiling/profiler.hpp"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tflite_model_parameters/tflite_model_parameters.hpp"
#include "tflite_micro_model/tflite_micro_model_details.hpp"
#include "tflite_micro_model/tflite_micro_tensor.hpp"

#include "mltk_tflite_micro_helper.hpp"
#include "mltk_tflite_micro_recorded_data.hpp"



namespace mltk
{

class TfliteMicroModel
{
public:
    TfliteModelParameters parameters;

    /**
     * Default constructor
     */
    TfliteMicroModel() = default;

    /**
     * Cleanup any allocated data
     */
    ~TfliteMicroModel();

    /**
     * @brief Load model flatbuffer
     * 
     * Load a model flatbuffer (.tflite).
     * 
     * @note The provided `flatbuffer` MUST persist for the life of this model object.
     * 
     * @note If runtime_buffer is NULL then a buffer will be automatically allocated.
     * If runtime_buffer_size = 0 then attempt to retrieve the arena size from the .tflite parameters metadata.
     * 
     * @param flatbuffer Model flatbuffer (.tflite) binary data
     * @param op_resolver @ref tflite::MicroOpResolver with reigstered kernels
     * @param runtime_buffer Buffer to hold model working memory
     * @param runtime_buffer_size Size of the given runtime_buffer in bytes
     * @return true if model successfully loaded, false else
     */
    bool load(
      const void* flatbuffer, 
      tflite::MicroOpResolver& op_resolver,
      uint8_t *runtime_buffer = nullptr,
      unsigned runtime_buffer_size = 0 
    );

    /**
     * @brief Unload model
     * 
     * Unload model and clean up and allocated resources
     */
    void unload();

    /**
     * @brief Return if a model is loaded
     * 
     * @return Return true if a model was successfully loaded, false otherwise
     */
    bool is_loaded() const;

   /**
     * @brief Invoke model inference
     * 
     * Execute the loaded model
     * 
     * @return true if model executed successfully, false else
     */
    bool invoke() const;

    /**
     * @brief Return model details
     * 
     * @return @ref ModelDetails
     */
    const TfliteMicroModelDetails& details() const;

    /**
     * @brief Print a summary of the model
     * 
     * 
     * @param logger Optional, if provided then print using this logger 
     *               else print using the default logger.
     */
    void print_summary(logging::Logger *logger = nullptr) const;

    /**
     * @brief Return number of input tensors
     * 
     * @return The number of model input tensors
     */
    unsigned input_size() const;

    /**
     * @brief Get input tensor
     * 
     * Populate the provided @ref TfliteTensorView with the 
     * details of the input tensor at the given index.
     * 
     * @param index Optional, index of input tensor
     * @return @ref TfliteTensorView to populate with input tensor at `index`
     */
    TfliteTensorView* input(unsigned index = 0) const;

    /**
     * @brief Return number of output tensors
     * 
     * @return The number of model output tensors
     */
    unsigned output_size() const;

    /**
     * @brief Get output tensor
     * 
     * Populate the provided @ref TfliteTensorView with the 
     * details of the output tensor at the given index.
     * 
     * @param index Optional, index of output tensor
     * @return @ref TfliteTensorView to populate with output tensor at `index`
     */
    TfliteTensorView* output(unsigned index = 0) const;

    /**
     * @brief Find metadata with tag in flatbuffer
     * 
     * Find metadata with the given `tag` in the model's flatubffer
     * 
     * @param tag Tag to search for in flatbuffer's metadata
     * @param length Optional, pointer to hold length of metadata's binary data
     * @return Pointer to found metadata's buffer in flatbuffer, null if not found
     */
    const void* find_metadata(const char* tag, uint32_t* length = nullptr) const;


   /**
     * Enable profiling of the ML model
     * 
     * @note This must be called BEFORE the model is loaded
     * 
     * @return true if profiler enabled, false else
     */
    bool enable_profiler();

    /**
     * Return if profiling is enabled
     * 
     * @return true if profiler is enabled, false else
     */
    bool profiler_is_enabled() const;

    /** 
     * Return the model profiler
     * 
     * @return Pointer to this model's profiler instance
     */
    profiling::Profiler* profiler() const;

   /**
     * Enable recording of model tensors during inference
     * 
     * @note This must be called BEFORE the model is loaded
     * 
     * @return true if recorder enabled, false else
     */
    bool enable_recorder();

    /**
     * Return if profiling is enabled
     * 
     * @return true if profiler is enabled, false else
     */
    bool recording_is_enabled() const;

    /**
     * Return the recorded data from the previous inference
     */
    TfliteMicroRecordedData& recorded_data();

    /**
     * Return a pointer to the TfliteMicroErrorReporter
     * used by the model
     */
    TfliteMicroErrorReporter* error_reporter()
    {
      return &_error_reporter;
    }

    /**
     * Return a pointer to the TfliteMicroInterpreter
     * used by the model
     */
    tflite::MicroInterpreter* interpreter()
    {
      return _interpreter;
    }

    /**
     * Return a pointer to the tflite::MicroOpResolver
     * used by the model
     */
    tflite::MicroOpResolver* ops_resolver()
    {
      return _ops_resolver;
    }

    /**
     * Set a callback to be called at the start of model
     * inference (i.e start of model.invoke()) 
     * and at the end of each model layer.
     */
    void set_processing_callback(void (*callback)(void*), void *arg = nullptr);

private:
  uint8_t _interpreter_buffer[sizeof(tflite::MicroInterpreter)];
  tflite::MicroInterpreter* _interpreter = nullptr;
  tflite::MicroOpResolver* _ops_resolver = nullptr;
  TfliteMicroModelDetails _model_details;
  TfliteMicroErrorReporter _error_reporter;
  const void* _flatbuffer = nullptr;
  void (*_processing_callback)(void*) = nullptr;
  void* _processing_callback_arg = nullptr;
  uint8_t* _runtime_buffer = nullptr;

  bool load_interpreter(
      const void* flatbuffer, 
      tflite::MicroOpResolver& op_resolver,
      uint8_t *runtime_buffer,
      unsigned runtime_buffer_size 
  );
  bool find_optimal_buffer_size(
      const void* flatbuffer, 
      tflite::MicroOpResolver& op_resolver,
      unsigned &runtime_buffer_size 
  );
  bool load_model_parameters(const void* flatbuffer=nullptr);
};


} // namespace mltk


extern "C" void mltk_sl_tflite_micro_init(mltk::TfliteMicroModel *model);
extern "C" TfLiteStatus mltk_sl_tflite_micro_invoke();


