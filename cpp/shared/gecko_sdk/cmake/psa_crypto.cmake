
set(NAME "mltk_gecko_sdk_psa_crypto")
add_library(${NAME})
add_library(mltk::gecko_sdk::psa_crypto ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_include_directories(${NAME}
PUBLIC
    sl_component/sl_mbedtls_support/inc
    sl_component/sl_mbedtls_support/config
    sl_component/sl_psa_driver/inc
    sl_component/sl_psa_driver/inc/public
    sl_component/sl_protocol_crypto/src
)

target_link_options(${NAME}
PUBLIC
    -Wl,-usli_psa_context_get_size
)

target_sources(${NAME}
PRIVATE
    sl_component/sl_mbedtls_support/src/sl_mbedtls.c
    sl_component/sl_mbedtls_support/src/sli_psa_crypto.c
    sl_component/sl_protocol_crypto/src/sli_protocol_crypto_radioaes.c
    sl_component/sl_protocol_crypto/src/sli_radioaes_management.c
    sl_component/sl_psa_driver/src/sl_psa_its_nvm3.c
    sl_component/sl_psa_driver/src/sli_psa_driver_common.c
    sl_component/sl_psa_driver/src/sli_psa_driver_init.c
    sl_component/sl_psa_driver/src/sli_psa_trng.c
)


target_link_libraries(${NAME}
PUBLIC
    mltk::gecko_sdk::memory_manager
    mltk::gecko_sdk::mbedtls
PRIVATE
    mltk::gecko_sdk::nvm3
    ${GECKO_SDK_BOARD_TARGET}
)


mltk_get(GECKO_SDK_SECURE_ELEMENT_ENABLED)
if(GECKO_SDK_SECURE_ELEMENT_ENABLED)
    target_include_directories(${NAME}
    PUBLIC
        sl_component/se_manager/inc
    PRIVATE
        sl_component/se_manager/src
    )

    target_sources(${NAME}
    PRIVATE
        sl_component/se_manager/src/sl_se_manager.c
        sl_component/se_manager/src/sl_se_manager_attestation.c
        sl_component/se_manager/src/sl_se_manager_cipher.c
        sl_component/se_manager/src/sl_se_manager_entropy.c
        sl_component/se_manager/src/sl_se_manager_hash.c
        sl_component/se_manager/src/sl_se_manager_key_derivation.c
        sl_component/se_manager/src/sl_se_manager_key_handling.c
        sl_component/se_manager/src/sl_se_manager_signature.c
        sl_component/se_manager/src/sl_se_manager_util.c
        sl_component/sl_psa_driver/src/sli_se_driver_aead.c
        sl_component/sl_psa_driver/src/sli_se_driver_builtin_keys.c
        sl_component/sl_psa_driver/src/sli_se_driver_cipher.c
        sl_component/sl_psa_driver/src/sli_se_driver_key_derivation.c
        sl_component/sl_psa_driver/src/sli_se_driver_key_management.c
        sl_component/sl_psa_driver/src/sli_se_driver_mac.c
        sl_component/sl_psa_driver/src/sli_se_driver_signature.c
        sl_component/sl_psa_driver/src/sli_se_opaque_driver_aead.c
        sl_component/sl_psa_driver/src/sli_se_opaque_driver_cipher.c
        sl_component/sl_psa_driver/src/sli_se_opaque_driver_mac.c
        sl_component/sl_psa_driver/src/sli_se_opaque_key_derivation.c
        sl_component/sl_psa_driver/src/sli_se_transparent_driver_aead.c
        sl_component/sl_psa_driver/src/sli_se_transparent_driver_cipher.c
        sl_component/sl_psa_driver/src/sli_se_transparent_driver_hash.c
        sl_component/sl_psa_driver/src/sli_se_transparent_driver_mac.c
        sl_component/sl_psa_driver/src/sli_se_transparent_key_derivation.c
        sl_component/sl_psa_driver/src/sli_se_version_dependencies.c
    )

endif()