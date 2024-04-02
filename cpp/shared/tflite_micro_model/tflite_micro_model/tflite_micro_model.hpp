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
     * @note If runtime_buffer is NULL then a buffer will be automatically allocated. In this case,
     * - If runtime_buffer_size < 0, then automatically find the optimal run-time buffer size
     * - If runtime_buffer_size > 0, then allocate the specified size, if the size is too small then return the error
     * - If runtime_buffer_size == 0, then attempt to retrieve the arena size from the .tflite parameters metadata,
     *     and allocate the buffer. If the buffer is too small or was not found in the metadata, 
     *     then automatically find the optimal run-time buffer size.
     * 
     * @param flatbuffer Model flatbuffer (.tflite) binary data
     * @param op_resolver @ref tflite::MicroOpResolver with reigstered kernels
     * @param runtime_buffer Buffer to hold model working memory
     * @param runtime_buffer_size Size of the given runtime_buffer in bytes
     * @return true if model successfully loaded, false else
     */
    bool load(
      const void* flatbuffer, 
      const tflite::MicroOpResolver& op_resolver,
      uint8_t *runtime_buffer = nullptr,
      int32_t runtime_buffer_size = 0
    );

    /**
     * @brief Load model with multiple runtime memory buffers
     * 
     * Load a model flatbuffer (.tflite) with multiple runtime memory buffers.
     * The model must be pre-compiled to leverage the given memory buffers.
     * 
     * @note The provided `flatbuffer` MUST persist for the life of this model object.
     * 
     * The first entry in the buffer list is also used to store "persistent" memory
     * and any temporary allocations.
     * 
     * @note If buffer[0] is NULL then a buffer will be automatically allocated. In this case,
     * - If buffer_sizes[0] < 0, then automatically find the optimal run-time buffer size
     * - If buffer_sizes[0] > 0, then allocate the specified size, if the size is too small then return the error
     * - If buffer_sizes[0] == 0, then attempt to retrieve the arena size from the .tflite parameters metadata,
     *     and allocate the buffer. If the buffer is too small or was not found in the metadata, 
     *     then automatically find the optimal run-time buffer size.
     * 
     * @param flatbuffer Model flatbuffer (.tflite) binary data
     * @param op_resolver @ref tflite::MicroOpResolver with reigstered kernels
     * @param buffers List of buffers to hold the model's runtime buffers
     * @param buffer_sizes Size of each buffer
     * @param buffer_count Number of buffer provided
     * @return true if model successfully loaded, false else
     */
    bool load(
      const void* flatbuffer, 
      const tflite::MicroOpResolver* op_resolver,
      uint8_t* buffers[],
      const int32_t buffer_sizes[],
      int32_t buffer_count
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
     * Enable recording of model information.
     * 
     * @note This must be called BEFORE the model is loaded
     * 
     * @return true if recorder enabled, false else
     */
    bool enable_recorder();

    /**
     * Return if model recording is enabled
     * 
     * @return true if model recording is enabled, false else
     */
    bool is_recorder_enabled() const;

   /**
     * Enable recording of model tensors during inference
     * 
     * @note This must be called BEFORE the model is loaded
     * 
     * @return true if recorder enabled, false else
     */
    bool enable_tensor_recorder();

    /**
     * Return if tensor recording is enabled
     * 
     * @return true if tensor recording is enabled, false else
     */
    bool is_tensor_recorder_enabled() const;


    /**
     * Return the recorded data from the previous inference
     * The returned data is msgpack formatted.
     */
    bool recorded_data(const uint8_t** buffer_ptr, uint32_t* length_ptr) const;


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
    const tflite::MicroOpResolver* ops_resolver()
    {
      return _ops_resolver;
    }

    /**
     * Return a pointer to the TfliteContext of the MicroInterpreter instance
     * used by the model
     */
    TfLiteContext* tflite_context() const
    {
      return (_interpreter != nullptr) ? &_interpreter->context_ : nullptr;
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
  const tflite::MicroOpResolver* _ops_resolver = nullptr;
  TfliteMicroModelDetails _model_details;
  const void* _flatbuffer = nullptr;
  void (*_processing_callback)(void*) = nullptr;
  void* _processing_callback_arg = nullptr;
  uint8_t* _runtime_buffer = nullptr;

  bool load_interpreter(
      const void* flatbuffer, 
      const tflite::MicroOpResolver* op_resolver,
      uint8_t* buffers[],
      const int32_t buffer_sizes[],
      int32_t buffer_count
  );

  bool find_optimal_buffer_size(
      const void* flatbuffer, 
      const tflite::MicroOpResolver* op_resolver,
      uint8_t* buffers[],
      int32_t buffer_sizes[],
      int32_t buffer_count,
      int32_t &optimal_buffer_size
  );
  bool load_model_parameters(const void* flatbuffer=nullptr);
};


} // namespace mltk
