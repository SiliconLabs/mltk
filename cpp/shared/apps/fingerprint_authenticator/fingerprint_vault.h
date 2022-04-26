#pragma once 

#include <stdbool.h>
#include <stdint.h>


#ifdef __cplusplus
extern "C" {
#endif



typedef struct
{
    const int8_t* data;
    uint8_t length;
} FingerprintSignature;

typedef struct
{
    uint8_t signature_length;
} fingerprint_vault_config_t;



bool fingerprint_vault_init(const fingerprint_vault_config_t *config);
bool fingerprint_vault_iterate_user(
    int32_t user_id,
    void (*callback)(int32_t uid, int32_t fpid, const FingerprintSignature*, void* arg),
    void* arg
);
bool fingerprint_vault_save_user_signature(const FingerprintSignature* signature, int32_t user_id);
bool fingerprint_vault_erase_user_signatures(int32_t user_id);
int fingerprint_vault_count_user_signatures(int32_t user_id);


#ifdef __cplusplus
}
#endif