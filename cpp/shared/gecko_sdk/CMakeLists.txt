
project(mltk_gecko_sdk
VERSION 1.0.0
DESCRIPTION "Wrapper project for Silab's Gecko SDK"
HOMEPAGE_URL "https://github.com/SiliconLabs/gecko_sdk"
)
export(PACKAGE ${PROJECT_NAME})

include(${CMAKE_CURRENT_LIST_DIR}/gecko_sdk_options.cmake)


mltk_info("Processing Gecko SDK repository (this may take awhile) ...")
if(DEFINED ENV{GIT_LFS_SKIP_SMUDGE})
  set(saved_GIT_LFS_SKIP_SMUDGE $ENV{GIT_LFS_SKIP_SMUDGE})
endif()

# Do NOT download this repo's git LFS files as they can take a long time to download
set(ENV{GIT_LFS_SKIP_SMUDGE} 1)
CPMAddPackage(
  NAME gecko_sdk
  GITHUB_REPOSITORY SiliconLabs/gecko_sdk
  GIT_TAG 3abe1693dfe8f8968de38efe403cd26d138eeeed # Gecko SDK v4.1.3, be sure to update the <mltk repo root>/README.md when updating this
  CACHE_VERSION v4.1.3
  DOWNLOAD_ONLY ON
)

unset(ENV{GIT_LFS_SKIP_SMUDGE})
if(DEFINED saved_GIT_LFS_SKIP_SMUDGE)
  set(ENV{GIT_LFS_SKIP_SMUDGE} ${saved_GIT_LFS_SKIP_SMUDGE})
endif()


function(gecko_sdk_add src dst)
  set(src_path ${CMAKE_CURRENT_LIST_DIR}/cmake/${src}.cmake)
  set(dst_path ${gecko_sdk_SOURCE_DIR}/${dst}/CMakeLists.txt)
  mltk_copy(${dst_path} ${src_path})
  add_subdirectory(${gecko_sdk_SOURCE_DIR}/${dst} ${MLTK_BINARY_DIR}/shared/gecko_sdk/${dst})
endfunction()

mltk_get(GECKO_SDK_INCLUDE_EMLIB)
if(NOT DEFINED GECKO_SDK_INCLUDE_EMLIB OR GECKO_SDK_INCLUDE_EMLIB)
  gecko_sdk_add(emlib platform/emlib)
endif()

gecko_sdk_add(hardware_board hardware/board)
gecko_sdk_add(platform_common platform/common)
gecko_sdk_add(iostream_service platform/service/iostream)


mltk_get(MLTK_PLATFORM_IS_EMBEDDED)
if(MLTK_PLATFORM_IS_EMBEDDED)
  gecko_sdk_add(efm32gg11b platform/Device/SiliconLabs/EFM32GG11B)
  gecko_sdk_add(efr32mg24 platform/Device/SiliconLabs/EFR32MG24)
  gecko_sdk_add(efr32mg12p platform/Device/SiliconLabs/EFR32MG12P)

  gecko_sdk_add(system_service platform/service/system)
  gecko_sdk_add(power_manager_service platform/service/power_manager)
  gecko_sdk_add(sleeptimer_service platform/service/sleeptimer)
  gecko_sdk_add(device_init_service platform/service/device_init)
  gecko_sdk_add(hfxo_manager_service platform/service/hfxo_manager)
  gecko_sdk_add(udelay_service platform/service/udelay)
  gecko_sdk_add(leddrv platform/driver/leddrv)
  gecko_sdk_add(button platform/driver/button)
  gecko_sdk_add(dmadrv platform/emdrv/dmadrv)
  gecko_sdk_add(nvm3 platform/emdrv/nvm3)
  gecko_sdk_add(gpiointerrupt platform/emdrv/gpiointerrupt)
  gecko_sdk_add(rail platform/radio/rail_lib)
  gecko_sdk_add(platform_peripheral platform/peripheral)
  gecko_sdk_add(mic hardware/driver/mic)
  gecko_sdk_add(mx25_flash_shutdown hardware/driver/mx25_flash_shutdown)
  gecko_sdk_add(freertos util/third_party/freertos)
  gecko_sdk_add(psa_crypto util/third_party/crypto)
  gecko_sdk_add(memory_manager util/silicon_labs/silabs_core/memory_manager)
  gecko_sdk_add(bluetooth protocol/bluetooth)
endif()


add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/clib_wrappers)

add_library(mltk_gecko_sdk_includes INTERFACE)
add_library(mltk::gecko_sdk::includes ALIAS mltk_gecko_sdk_includes)
target_include_directories(mltk_gecko_sdk_includes
INTERFACE
  ${gecko_sdk_SOURCE_DIR}/hardware/driver/mic/inc
  ${gecko_sdk_SOURCE_DIR}/hardware/driver/button/inc
  ${gecko_sdk_SOURCE_DIR}/platform/driver/leddrv/inc
  ${gecko_sdk_SOURCE_DIR}/platform/common/inc 
  ${gecko_sdk_SOURCE_DIR}/platform/emdrv/dmadrv/inc
  ${gecko_sdk_SOURCE_DIR}/platform/emdrv/spidrv/inc
  ${gecko_sdk_SOURCE_DIR}/platform/emdrv/gpiointerrupt/inc
  ${gecko_sdk_SOURCE_DIR}/platform/emdrv/nvm3/inc
  ${gecko_sdk_SOURCE_DIR}/platform/service/system/inc
  ${gecko_sdk_SOURCE_DIR}/platform/service/sleeptimer/inc
  ${gecko_sdk_SOURCE_DIR}/platform/service/udelay/inc
)

