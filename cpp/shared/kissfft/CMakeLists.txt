project(mltk_kissfft
        VERSION 1.0.0
        DESCRIPTION "MLTK KissFFT library")
export(PACKAGE ${PROJECT_NAME})
add_library(${PROJECT_NAME})
add_library(mltk::kissfft ALIAS ${PROJECT_NAME})


# URL comes from:
# https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/tools/make/third_party_downloads.inc
CPMAddPackage(
    NAME kissfft
    URL  https://github.com/mborgerding/kissfft/archive/refs/tags/v130.zip
    URL_HASH MD5=438ba1fef5783cc5f5f201395cc477ca
    CACHE_VERSION v1.3.0
)
set(kissfft_SOURCE_DIR ${kissfft_SOURCE_DIR} CACHE INTERNAL "")
    

target_include_directories(${PROJECT_NAME}
PUBLIC 
    ${kissfft_SOURCE_DIR}
)	 
target_sources(${PROJECT_NAME}
PRIVATE 
    ${kissfft_SOURCE_DIR}/kiss_fft.c 
    ${kissfft_SOURCE_DIR}/tools/kiss_fftr.c
)
target_compile_options(${PROJECT_NAME}
PRIVATE 
    -Wno-nonnull
)
target_compile_definitions(${PROJECT_NAME}
PUBLIC 
    FIXED_POINT=16
)
target_compile_options(${PROJECT_NAME}
PRIVATE 
    -include stdint.h
)



mltk_load_python()

# Ensure the downloaded library is patched 
add_custom_command(OUTPUT ${kissfft_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
  DEPENDS ${kissfft_SOURCE_DIR}/tools ${CMAKE_CURRENT_LIST_DIR}/patch_kissfft.py
  COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_UTILS_DIR}/libpatcher.py -i ${kissfft_SOURCE_DIR}/tools -p ${CMAKE_CURRENT_LIST_DIR}/patch_kissfft.py -o ${kissfft_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt
)
add_custom_target(${PROJECT_NAME}_apply_patch DEPENDS ${kissfft_SOURCE_DIR}/${PROJECT_NAME}_patch_complete.txt)
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_apply_patch)
