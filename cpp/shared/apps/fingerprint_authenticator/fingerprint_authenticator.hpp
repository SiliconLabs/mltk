#pragma once 

#include <cstdint>

#include "logging/logging.hpp"
#include "tflite_micro_model/tflite_micro_model.hpp"
#include "tflite_micro_model/tflite_micro_tensor.hpp"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "fingerprint_reader/fingerprint_reader.h"

#include "fingerprint_vault.h"
#include "data_preprocessor.hpp"


namespace mltk
{


class FingerprintAuthenticator
{
public:
    TfliteMicroModel model;

    FingerprintAuthenticator() = default;

    bool init();
    bool load_model(const void* flatbuffer);

    bool generate_signature(
        const fingerprint_reader_image_t fingerprint_image, 
        FingerprintSignature& signature,
        bool &fingerprint_image_valid
    );

    bool authenticate_signature(
        const FingerprintSignature &signature, 
        int32_t &user_id
    );
    bool save_signature(
        const FingerprintSignature &signature, 
        int32_t user_id
    );
    bool remove_signatures(int32_t user_id);


    static const char* signature_to_str(const FingerprintSignature& signature, char* buffer = nullptr);
    

private:
    tflite::AllOpsResolver _op_resolver;
    DataPreprocessor _data_preprocessor;
    uint8_t* _tensor_arena = nullptr;
    TfliteTensorView* _input_tensor = nullptr;
    TfliteTensorView* _output_tensor = nullptr;
    uint8_t _signature_length = 0;
    float _auth_threshold = 0;
};

} // namespace mltk