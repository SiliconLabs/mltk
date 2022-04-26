#include <string.h>

#include "fingerprint_vault.h"
#include "app_config.hpp"
#include "nvm3_default.h"
#include "nvm3_default_config.h"



#define MAKE_NVM_KEY(user_id, index) ((((uint32_t)user_id) << 16) | ((uint32_t)index))


static uint8_t signature_length = 0;




/*************************************************************************************************/
static int retrieve_user_signature_keys(int32_t user_id, nvm3_ObjectKey_t *buffer)
{
    return nvm3_enumObjects(
        nvm3_defaultHandle, buffer, MAX_SIGNATURES_PER_USER, 
        MAKE_NVM_KEY(user_id, 0), MAKE_NVM_KEY(user_id, MAX_SIGNATURES_PER_USER-1)
    );
}

/*************************************************************************************************/
bool fingerprint_vault_init(const fingerprint_vault_config_t *config)
{
    Ecode_t err = nvm3_initDefault();
    if(err != ECODE_NVM3_OK)
    {
        return false;
    }

    signature_length = config->signature_length;

    if(nvm3_repackNeeded(nvm3_defaultHandle)) 
    {
        nvm3_repack(nvm3_defaultHandle);
    }
    return true;
}


/*************************************************************************************************/
bool fingerprint_vault_iterate_user(
    int32_t user_id,
    void (*callback)(int32_t uid, int32_t fpid, const FingerprintSignature*, void* arg),
    void* arg
)
{
    int32_t uid = (user_id == -1) ? 0 : user_id;
    const int32_t uid_end =  (user_id == -1) ? MAX_USERS : uid+1;

    for(; uid < uid_end; ++uid)
    {
        nvm3_ObjectKey_t signatures_keys[MAX_SIGNATURES_PER_USER];
        int sig_count = retrieve_user_signature_keys(uid, signatures_keys);
        for(int fpip = 0; fpip < sig_count; ++fpip)
        {
            int8_t signature_buffer[MAX_SIGNATURE_SIZE];

            Ecode_t err = nvm3_readData(
                nvm3_defaultHandle, 
                signatures_keys[fpip], 
                signature_buffer, 
                signature_length
            );
            if(err != ECODE_NVM3_OK)
            {
                continue;
            }

            const FingerprintSignature found_sig = 
            {
                signature_buffer, signature_length
            };

            callback(uid, fpip, &found_sig, arg);
        }
    }

    return true;
}


/*************************************************************************************************/
bool fingerprint_vault_save_user_signature(const FingerprintSignature* signature, int32_t user_id)
{
    nvm3_ObjectKey_t signatures_keys[MAX_SIGNATURES_PER_USER];
    int signature_count = retrieve_user_signature_keys(user_id, signatures_keys);
    Ecode_t result = nvm3_writeData(
        nvm3_defaultHandle, 
        MAKE_NVM_KEY(user_id, signature_count),
        signature->data, signature->length
    );

    return result == ECODE_NVM3_OK;
}

/*************************************************************************************************/
bool fingerprint_vault_erase_user_signatures(int32_t user_id)
{
    nvm3_ObjectKey_t signatures_keys[MAX_SIGNATURES_PER_USER];
    int signature_count = retrieve_user_signature_keys(user_id, signatures_keys);

    for(int i = 0; i < signature_count; ++i)
    {
        nvm3_deleteObject(nvm3_defaultHandle, signatures_keys[i]);
    }

    return true;
}

/*************************************************************************************************/
int fingerprint_vault_count_user_signatures(int32_t user_id)
{
    nvm3_ObjectKey_t signatures_keys[MAX_SIGNATURES_PER_USER];

    return nvm3_enumObjects(
        nvm3_defaultHandle, signatures_keys, MAX_SIGNATURES_PER_USER, 
        MAKE_NVM_KEY(user_id, 0), MAKE_NVM_KEY(user_id, MAX_SIGNATURES_PER_USER-1)
    );
}
