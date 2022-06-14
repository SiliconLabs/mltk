
#include <cstdio>
#include <cmath>
#include "app_config.hpp"
#include "tflite_micro_model/tflite_micro_tensor.hpp"
#include "tflite_micro_model/tflite_micro_utils.hpp"
#include "fingerprint_authenticator.hpp"
#include "fingerprint_vault.h"



namespace mltk
{

/*************************************************************************************************/
bool FingerprintAuthenticator::init()
{
    return true;
}

/*************************************************************************************************/
bool FingerprintAuthenticator::load_model(const void* flatbuffer)
{
    // Register the accelerator if the TFLM lib was built with one
    mltk_tflite_micro_register_accelerator();


    if(!model.load(flatbuffer, _op_resolver))
    {
        MLTK_ERROR("Failed to load .tflite");
        return false;
    }

    _input_tensor = model.input(0);
    _output_tensor = model.output(0);

    _signature_length = _output_tensor->shape().flat_size();

    const auto input_shape = _input_tensor->shape();
    const uint16_t input_width = input_shape[1];
    const uint16_t input_height = input_shape[2];

    model.parameters.get("disable_inference", _disabled);
    model.parameters.get("threshold", _auth_threshold);
    MLTK_INFO("Signature threshold: %.3f", _auth_threshold);

    if(!_data_preprocessor.load(model.parameters, input_width, input_height))
    {
        MLTK_ERROR("Failed to load data preprocessor");
        return false;
    }

    const fingerprint_vault_config_t vault_config = 
    {
        _signature_length
    };

    if(!fingerprint_vault_init(&vault_config))
    {
        MLTK_ERROR("Failed to initialize the fingerprint vault");
        return false;
    }


    return true;
}


/*************************************************************************************************/
bool FingerprintAuthenticator::generate_signature(
    const fingerprint_reader_image_t fingerprint_image, 
    FingerprintSignature& signature,
    bool &fingerprint_image_valid
)
{
    // If inference is disabled,
    // then just return
    if(_disabled)
    {
        fingerprint_image_valid = false;
        return true;
    }

    fingerprint_image_valid = true;
    int8_t* processed_image_buffer = _input_tensor->data.int8;
    const uint32_t processed_image_size = _input_tensor->shape().flat_size();


    if(!_data_preprocessor.preprocess_sample((const uint8_t*)fingerprint_image, (uint8_t*)processed_image_buffer))
    {
        MLTK_ERROR("Failed to preprocess sample");
        return false;
    }
 
    if(!_data_preprocessor.verify_sample((uint8_t*)processed_image_buffer))
    {
        MLTK_WARN("Image not valid");
        fingerprint_image_valid = false;
        return true;
    }

    int8_t *p = processed_image_buffer;
    for(int i = processed_image_size; i > 0; --i)
    {
        uint8_t v = *(uint8_t*)p;
        *p++ = (int8_t)(v - 128);
    }

    if(!model.invoke())
    {
        MLTK_ERROR("Failed to invoke ML model");
        return false;
    }

    signature.data = _output_tensor->data.int8;
    signature.length = _signature_length;

    return true;
}


/*************************************************************************************************/
bool FingerprintAuthenticator::authenticate_signature(
    const FingerprintSignature &signature, 
    int32_t &user_id
)
{
    struct CompareContext
    {
        const FingerprintSignature& signature;
        const TfLiteQuantizationParams& q_params;
        float best_score;
        int32_t best_user_id ;

        CompareContext(const FingerprintSignature& sig, const TfLiteQuantizationParams& q_params): 
            signature(sig), 
            q_params(q_params),
            best_score(1e6f),
            best_user_id(-1){}
    };


    auto compare_callback = [](int32_t uid, int32_t fpid, const FingerprintSignature* sig, void* ctx) 
    {
        CompareContext& context = *(CompareContext*)ctx;
        const int sig_length = sig->length;

        float sum = 0;
        for(int i = 0; i < sig_length; ++i)
        {
            const float s1 = dequantized_value(context.q_params, context.signature.data[i]);
            const float s2 = dequantized_value(context.q_params, sig->data[i]);
            const float diff = s1 - s2;
            sum += diff*diff;
        }
        const float score = sqrtf(sum);

        MLTK_INFO("User: %d-%d similarity score: %.3f", uid, fpid, score);
        if(score < context.best_score)
        {
            context.best_score = score;
            context.best_user_id = uid;
        }
    };

    CompareContext context(signature, _output_tensor->params);
    user_id = -1;

    if(!fingerprint_vault_iterate_user(-1, compare_callback, (void*)&context))
    {
        return false;
    }

    if(context.best_score < _auth_threshold)
    {
        user_id = context.best_user_id;
    }

    return true;
}


/*************************************************************************************************/
bool FingerprintAuthenticator::save_signature(
    const FingerprintSignature &signature, 
    int32_t user_id
)
{
    return fingerprint_vault_save_user_signature(&signature, user_id);
}


/*************************************************************************************************/
bool FingerprintAuthenticator::remove_signatures(int32_t user_id)
{
    return fingerprint_vault_erase_user_signatures(user_id);
}


/*************************************************************************************************/
const char* FingerprintAuthenticator::signature_to_str(const FingerprintSignature& signature, char* buffer)
{
    static char signature_str[MAX_SIGNATURE_SIZE*4 + 2];

    buffer = (buffer == nullptr) ? signature_str : buffer;
    char *s = buffer;

    if(signature.data == nullptr)
    {
        strcpy(buffer, "null");
    }
    else if(signature.length > MAX_SIGNATURE_SIZE)
    {
        strcpy(buffer, "overflow");
    }
    else 
    {
        *s = 0;
        for(int i = 0; i < signature.length; ++i)
        {
            s += sprintf(s, "%d ", signature.data[i]);
        }
    }

    return buffer;
}



} // namespace mltk