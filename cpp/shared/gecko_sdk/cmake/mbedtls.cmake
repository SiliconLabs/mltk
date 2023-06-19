
set(NAME "mltk_gecko_sdk_mbedtls")
add_library(${NAME})
add_library(mltk::gecko_sdk::mbedtls ALIAS ${NAME})


mltk_get(GECKO_SDK_BOARD_TARGET)
if(NOT GECKO_SDK_BOARD_TARGET)
    mltk_error("Must specify GECKO_SDK_BOARD_TARGET global property")
endif()


target_include_directories(${NAME}
PUBLIC
    include
    library
)

target_compile_definitions(${NAME}
PUBLIC
    MBEDTLS_CONFIG_FILE=<sl_mbedtls_config.h>
    MBEDTLS_PSA_CRYPTO_CONFIG_FILE=<psa_crypto_config.h>
)

target_sources(${NAME}
PRIVATE
    library/cipher.c
    library/cipher_wrap.c
    library/constant_time.c
    library/platform.c
    library/platform_util.c
    library/psa_crypto.c
    library/psa_crypto_aead.c
    library/psa_crypto_cipher.c
    library/psa_crypto_client.c
    library/psa_crypto_driver_wrappers.c
    library/psa_crypto_ecp.c
    library/psa_crypto_hash.c
    library/psa_crypto_mac.c
    library/psa_crypto_rsa.c
    library/psa_crypto_slot_management.c
    library/psa_crypto_storage.c
    library/threading.c
)


target_link_libraries(${NAME}
PUBLIC
    mltk::gecko_sdk::memory_manager
PRIVATE
    mltk::gecko_sdk::nvm3
    ${GECKO_SDK_BOARD_TARGET}
)


mltk_get(GECKO_SDK_SECURE_ELEMENT_ENABLED)
if(GECKO_SDK_SECURE_ELEMENT_ENABLED)
    target_sources(${NAME}
    PRIVATE
        library/psa_crypto_se.c
    )

endif()