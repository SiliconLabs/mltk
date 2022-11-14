# Name of the target

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(TOOLCHAIN_PREFIX arm-none-eabi)
set(MLTK_TOOLCHAIN_NAME arm CACHE INTERNAL "")


include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/utilities.cmake)

string(REGEX REPLACE "/CMakeFiles/CMakeTmp$" "" _bin_dir ${CMAKE_BINARY_DIR})

mltk_info("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
if(NOT MLTK_USER_OPTIONS)
  set(MLTK_USER_OPTIONS ${CMAKE_SOURCE_DIR}/user_options.cmake)
endif()

if(EXISTS "${MLTK_USER_OPTIONS}" AND NOT MLTK_NO_USER_OPTIONS)
  include("${MLTK_USER_OPTIONS}")
endif()


mltk_get(MLTK_PLATFORM_NAME)
if(MLTK_PLATFORM_NAME AND MLTK_PREVIOUS_PLATFORM_NAME)
    if(NOT "${MLTK_PLATFORM_NAME}" STREQUAL "${MLTK_PREVIOUS_PLATFORM_NAME}")
        message(NOTICE "\n*************************************************************************************************************\n")
        message(NOTICE "Current MLTK_PLATFORM_NAME=${MLTK_PLATFORM_NAME} and MLTK_PREVIOUS_PLATFORM_NAME=${MLTK_PREVIOUS_PLATFORM_NAME}")
        message(NOTICE "Cleaning build directory: ${_bin_dir}")
        message(NOTICE "\nYou must re-configure your CMake project to continue!!")
        message(NOTICE "\n************************************************************************************************************\n\n")

        file(REMOVE_RECURSE "${_bin_dir}")
        get_cmake_property(_variableNames VARIABLES)
        foreach (_variableName ${_variableNames})
            unset(${_variableName} CACHE)
        endforeach()

        return()
    endif()
endif()



if(NOT TOOLCHAIN_DIR)
    set(gcc_path_file "${_bin_dir}/gcc_path.txt")
    if(NOT EXISTS ${gcc_path_file})
        mltk_load_python()
        message(NOTICE "Preparing GCC ARM toolchain (this may take awhile) ...")
        execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/download_toolchain.py --noprogress RESULT_VARIABLE result OUTPUT_VARIABLE output)
        if(result) 
            if(output)
                list(GET output 0 _error_file_path)
                if(EXISTS "${_error_file_path}")
                    file (STRINGS ${_error_file_path} _err_msg)
                    string(REPLACE ";" "\n" _err_msg ${_err_msg})
                endif()
            endif()
            unset(PYTHON_EXECUTABLE CACHE)
            file(REMOVE "${_bin_dir}/CMakeCache.txt")
            file(REMOVE_RECURSE "${_bin_dir}/CMakeFiles")
            message(FATAL_ERROR "${_err_msg}\nFailed to download GCC ARM toolchain, see: ${CMAKE_CURRENT_LIST_DIR}/download.log\nAlso see: https://siliconlabs.github.io/mltk/docs/cpp_development/index.html\n\n")
        endif()

        list(GET output 0 TOOLCHAIN_DIR)
        list(GET output 1 TOOLCHAIN_BIN_EXT)
        file(WRITE ${gcc_path_file} "${output}")
    else()
        file(READ ${gcc_path_file} _data)
        list(GET _data 0 TOOLCHAIN_DIR)
        list(GET _data 1 TOOLCHAIN_BIN_EXT)
    endif()
endif()

if(NOT CMAKE_MAKE_PROGRAM AND "${CMAKE_GENERATOR}" STREQUAL "Ninja")
  set(ninja_path_file "${_bin_dir}/ninja_path.txt")
  if(NOT EXISTS ${ninja_path_file})
    mltk_load_python()
    set(_ninja_dir ${CMAKE_CURRENT_LIST_DIR}/../../../utils)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} ${_ninja_dir}/get_ninja_path.py RESULT_VARIABLE result OUTPUT_VARIABLE output)
    list(GET output 0 CMAKE_MAKE_PROGRAM)
    if(result) 
      message(FATAL_ERROR "Failed to get path to Ninja executable: ${CMAKE_MAKE_PROGRAM}")
    endif()
    file(WRITE ${ninja_path_file} ${CMAKE_MAKE_PROGRAM})
  else()
    file(READ ${ninja_path_file} CMAKE_MAKE_PROGRAM)
  endif()
endif()

# Toolchain settings

set(TOOLCHAIN_DIR       ${TOOLCHAIN_DIR} CACHE INTERNAL "ARM GCC toolchain directory")
set(TOOLCHAIN_BIN_EXT   ${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "ARM GCC toolchain executable extension")
set(TOOLCHAIN_BIN_DIR   ${TOOLCHAIN_DIR}/bin)
set(CMAKE_C_COMPILER    ${TOOLCHAIN_BIN_DIR}/${TOOLCHAIN_PREFIX}-gcc${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "c compile")
set(CMAKE_CXX_COMPILER  ${TOOLCHAIN_BIN_DIR}/${TOOLCHAIN_PREFIX}-g++${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "c++ compiler")
set(CMAKE_AR            ${TOOLCHAIN_BIN_DIR}/${TOOLCHAIN_PREFIX}-ar${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "")
set(CMAKE_AS            ${TOOLCHAIN_BIN_DIR}/${TOOLCHAIN_PREFIX}-as${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "")
set(CMAKE_OBJCOPY       ${TOOLCHAIN_BIN_DIR}/${TOOLCHAIN_PREFIX}-objcopy${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "")
set(CMAKE_OBJDUMP       ${TOOLCHAIN_BIN_DIR}/${TOOLCHAIN_PREFIX}-objdump${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "")
set(CMAKE_SIZE          ${TOOLCHAIN_BIN_DIR}/${TOOLCHAIN_PREFIX}-size${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "")
set(CMAKE_STRIP         ${TOOLCHAIN_BIN_DIR}/${TOOLCHAIN_PREFIX}-strip${TOOLCHAIN_BIN_EXT} CACHE INTERNAL "")
set(CMAKE_MAKE_PROGRAM  ${CMAKE_MAKE_PROGRAM} CACHE INTERNAL "Ninja generation program")
set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR})
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})


set(SPEC_PATH "${TOOLCHAIN_DIR}/arm-none-eabi/lib")
set(CLIB_SPECS "--specs ${SPEC_PATH}/nano.specs --specs ${SPEC_PATH}/nosys.specs")

set(CMAKE_C_FLAGS_INIT   "${CLIB_SPECS} -nostartfiles -ffunction-sections -fdata-sections -ffreestanding -fno-common -fno-delete-null-pointer-checks -Wno-unused-parameter -fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=/ -fmacro-prefix-map=${MLTK_CPP_DIR}/=/" CACHE INTERNAL "c compiler flags")
set(CMAKE_CXX_FLAGS_INIT "${CLIB_SPECS} -nostartfiles -ffunction-sections -fdata-sections -ffreestanding -fno-common -fno-delete-null-pointer-checks -Wno-unused-parameter -fno-threadsafe-statics -fno-rtti -fno-exceptions -fno-use-cxa-atexit -fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=/ -fmacro-prefix-map=${MLTK_CPP_DIR}/=/" CACHE INTERNAL "cxx compiler flags")
set(CMAKE_ASM_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} -x assembler-with-cpp")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--gc-sections -Wl,--cref" CACHE INTERNAL "exe link flags")

# NOTE: Using -ggdb instead of -ggdb3 due to a bug in the VSCode cortex-debug extension:
# https://www.cygwin.com/bugzilla/show_bug.cgi?id=28219
SET(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb -DDEBUG" CACHE INTERNAL "c debug compiler flags")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG" CACHE INTERNAL "cxx debug compiler flags")
SET(CMAKE_ASM_FLAGS_DEBUG "-ggdb" CACHE INTERNAL "asm debug compiler flags")


mltk_get(MLTK_ENABLE_DEBUG_INFO_IN_RELEASE_BUILDS)
if(MLTK_ENABLE_DEBUG_INFO_IN_RELEASE_BUILDS)
SET(CMAKE_C_FLAGS_RELEASE "-O3 -g -ggdb" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ggdb" CACHE INTERNAL "cxx release compiler flags")
else()
SET(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE INTERNAL "cxx release compiler flags")
endif()

set(CMAKE_EXE_LINKER_RELEASE "-flto" CACHE INTERNAL "exe link flags")

SET(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb" CACHE INTERNAL "cxx release compiler flags")
set(CMAKE_EXE_LINKER_RELWITHDEBINFO "-flto" CACHE INTERNAL "exe link flags")

SET(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1 CACHE INTERNAL "")
SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1 CACHE INTERNAL "")

SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@" CACHE INTERNAL "")
SET(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@" CACHE INTERNAL "")

SET(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(MLTK_SIZE_OPTIMIZATION_FLAG -O0 CACHE INTERNAL "")
else()
    set(MLTK_SIZE_OPTIMIZATION_FLAG -Os CACHE INTERNAL "")
endif()

# Don't register CTests for embedded target, since cannot execute directly
SET(MLTK_EXCLUDE_TESTS ON CACHE INTERNAL "Exclude MLTK CTest generation")

######################################################################
# mltk_toolchain_add_exe_targets
#
# Generate the executions output files. These are files that may be
# downloaded to the embedded device.
#
# target - The executabe CMake build target
macro(mltk_toolchain_add_exe_targets target)
    set(_output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    set(_output_path ${_output_dir}/${target}.elf)
    set(_output_prefix ${_output_dir}/${target})

    mltk_load_python()
    mltk_get(MLTK_PLATFORM_NAME)

    set_target_properties(${target}
    PROPERTIES 
        SUFFIX ".elf"
    )

    target_link_options(${target}
    PUBLIC 
      -Wl,-Map,${_output_prefix}.map
    )

    mltk_get(MLTK_STRIP_ELF_SECTIONS)
    if(MLTK_STRIP_ELF_SECTIONS)
        mltk_info("Stripping sections from .elf: ${MLTK_STRIP_ELF_SECTIONS}")
        set(_strip_args)
        foreach(section ${MLTK_STRIP_ELF_SECTIONS})
            set(_strip_args "${_strip_args} -R ${section}")
        endforeach()
    endif()

    mltk_get(MLTK_PLATFORM_NAME)
    mltk_get(MLTK_JLINK_DEVICE)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/post_build.cmake 
"
execute_process(COMMAND ${CMAKE_STRIP} ${_strip_args} -o \"${_output_prefix}.stripped.elf\" \"${_output_path}\")
execute_process(COMMAND ${CMAKE_OBJCOPY} -R .bootloader -O binary \"${_output_prefix}.stripped.elf\" \"${_output_prefix}.bin\")
execute_process(COMMAND ${CMAKE_OBJCOPY} -j .bootloader -O binary \"${_output_prefix}.stripped.elf\" \"${_output_prefix}.bootloader.bin\")
execute_process(COMMAND ${CMAKE_OBJCOPY} -O srec \"${_output_prefix}.stripped.elf\" \"${_output_prefix}.s37\")
execute_process(COMMAND ${CMAKE_SIZE} \"${_output_path}\")
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_DIR}/tools/elf-size-analyze/elf-size-analyze.py \"${_output_path}\" --toolchain-path ${TOOLCHAIN_BIN_DIR}/arm-none-eabi- --no-color --ram --rom --human-readable --max-width 120 --output \"${_output_dir}/${target}-memory_usage.txt\")
execute_process(COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_DIR}/tools/utils/update_launch_json.py --name ${target} --path \"${_output_path}\" --toolchain \"${TOOLCHAIN_BIN_DIR}\" --platform ${MLTK_PLATFORM_NAME} --workspace \"${CMAKE_SOURCE_DIR}\" --jlink_device \"${MLTK_JLINK_DEVICE}\" )
execute_process(COMMAND ${CMAKE_COMMAND} -E echo \"For more details, see: ${_output_dir}/${target}-memory_usage.txt\")
")

    add_custom_command(TARGET ${target}
        POST_BUILD 
        COMMAND ${CMAKE_COMMAND} -P post_build.cmake
        BYPRODUCTS 
            "${_output_prefix}.stripped.elf"
            "${_output_prefix}.bin"
            "${_output_prefix}.bootloader.bin"
            "${_output_prefix}.s37"
            "${_output_dir}/${target}-memory_usage.txt"
    )

    unset(_output_path)
    unset(_output_prefix)
    unset(_output_dir)

endmacro()

