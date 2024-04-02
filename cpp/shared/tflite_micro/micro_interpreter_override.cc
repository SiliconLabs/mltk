
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

#include "mltk_tflite_micro_helper.hpp"


namespace tflite
{
TfLiteStatus MicroInterpreter::PrepareNodeAndRegistrationDataFromFlatbuffer() {
  TfLiteStatus retval = kTfLiteOk;
  auto& logger = mltk::get_logger();

  for (int subgraph_idx = 0; subgraph_idx < graph_.NumSubgraphs();
       subgraph_idx++) {
    const SubGraph* subgraph = model_->subgraphs()->Get(subgraph_idx);
    TFLITE_DCHECK(subgraph != nullptr);

    auto* opcodes = model_->operator_codes();
    TfLiteBridgeBuiltinDataAllocator* builtin_data_allocator =
        allocator_.GetBuiltinDataAllocator();
    uint32_t operators_size = NumSubgraphOperators(subgraph);
    for (size_t i = 0; i < operators_size; ++i) {
      const auto* op = subgraph->operators()->Get(i);
      const size_t index = op->opcode_index();
      if (index >= opcodes->size()) {
        logger.error("Missing registration for opcode_index %d", index);
        retval = kTfLiteError;
        continue;
      }
      const auto* opcode = opcodes->Get(index);
      TfLiteStatus status =
          GetRegistrationFromOpCode(opcode, op_resolver_,
                                    &(graph_.GetAllocations()[subgraph_idx]
                                          .node_and_registrations[i]
                                          .registration));
      if (status != kTfLiteOk) {
        retval = kTfLiteError;
        logger.error("Op%d-%s not supported: Unknown layer type",
                    i, EnumNameBuiltinOperator(GetBuiltinCode(opcode)));
        continue;
      }
      const auto* registration = graph_.GetAllocations()[subgraph_idx]
                                     .node_and_registrations[i]
                                     .registration;
      if (registration == nullptr) {
        retval = kTfLiteError;
        logger.error("Op%d-%s not supported: Failed to get registration for opcode_index %d", i, EnumNameBuiltinOperator(GetBuiltinCode(opcode)), index);
        continue;
      }
      BuiltinOperator op_type =
          static_cast<BuiltinOperator>(registration->builtin_code);

      const char* custom_data = nullptr;
      size_t custom_data_size = 0;
      unsigned char* builtin_data = nullptr;

      if (op_type == BuiltinOperator_CUSTOM) {
        // Custom Ops may or may not have a non-null custom_options field.
        if (op->custom_options() != nullptr) {
          custom_data =
              reinterpret_cast<const char*>(op->custom_options()->data());
          custom_data_size = op->custom_options()->size();
        }
      } else {
        if (op->custom_options() != nullptr) {
          retval = kTfLiteError;
          logger.error(
              "Op%d-%s not supported: Found builtin operator %s with custom options",
              i, EnumNameBuiltinOperator(op_type));
          continue;
        }

        TfLiteBridgeBuiltinParseFunction parser =
            op_resolver_.GetOpDataParser(op_type);
        if (parser == nullptr) {
          retval = kTfLiteError;
          logger.error("Op%d-%s not supported: Did not find a parser",
                      i, EnumNameBuiltinOperator(op_type));

           continue;
        }
        TF_LITE_ENSURE_STATUS(CallBuiltinParseFunction(
            parser, op, builtin_data_allocator, (void**)(&builtin_data)));
      }

      TfLiteIntArray* inputs_array =
          FlatBufferVectorToTfLiteTypeArray(op->inputs());
      TfLiteIntArray* outputs_array =
          FlatBufferVectorToTfLiteTypeArray(op->outputs());

      TfLiteNode* node = &(
          graph_.GetAllocations()[subgraph_idx].node_and_registrations[i].node);
      *node = {};
      node->inputs = inputs_array;
      node->outputs = outputs_array;
      node->builtin_data = reinterpret_cast<void*>(builtin_data);
      node->custom_initial_data = custom_data;
      node->custom_initial_data_size = custom_data_size;

      if (op->intermediates() && (op->intermediates()->size() > 0)) {
        node->intermediates =
            FlatBufferVectorToTfLiteTypeArray(op->intermediates());
      }
    }
  }

  if(retval != kTfLiteOk)
  {
    mltk::TfliteMicroKernelMessages::set_unknown_layers_detected(true);
  }
  return retval;
}



} // namespace tflite