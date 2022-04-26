# Name of the target
set(CMAKE_SYSTEM_NAME Linux-GNU)
set(TOOLCHAIN_PREFIX x86_64-linux-gnu)
set(CMAKE_SYSTEM_PROCESSOR GCC)

set(MLTK_TOOLCHAIN_NAME linux CACHE INTERNAL "")


include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/utilities.cmake)

execute_process(
  COMMAND ${TOOLCHAIN_PREFIX}-gcc-8 --version 
  RESULT_VARIABLE result
  OUTPUT_VARIABLE __dummy
)
if(result)
  message(FATAL_ERROR "gcc8 must be installed and available on the exeuctable PATH. \
  Run the command: \
  sudo apt-get -y install build-essential g++-8 gdb \
  ")
endif()

if(NOT CMAKE_MAKE_PROGRAM AND "${CMAKE_GENERATOR}" STREQUAL "Ninja")
  string(REGEX REPLACE "/CMakeFiles/CMakeTmp$" "" _bin_dir ${CMAKE_BINARY_DIR})
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
set(CMAKE_C_COMPILER    ${TOOLCHAIN_PREFIX}-gcc-8 CACHE INTERNAL "")
set(CMAKE_CXX_COMPILER  ${TOOLCHAIN_PREFIX}-g++-8 CACHE INTERNAL "")
set(CMAKE_AR            ${TOOLCHAIN_PREFIX}-gcc-ar-8 CACHE INTERNAL "")
set(CMAKE_AS            ${TOOLCHAIN_PREFIX}-as CACHE INTERNAL "")
set(CMAKE_OBJCOPY       ${TOOLCHAIN_PREFIX}-uobjcopy CACHE INTERNAL "")
set(CMAKE_OBJDUMP       ${TOOLCHAIN_PREFIX}-objdump CACHE INTERNAL "")
set(CMAKE_SIZE          ${TOOLCHAIN_PREFIX}-size CACHE INTERNAL "")
set(CMAKE_READELF       ${TOOLCHAIN_PREFIX}-readelf  CACHE INTERNAL "")
set(CMAKE_STRIP         ${TOOLCHAIN_PREFIX}-strip  CACHE INTERNAL "")
set(CMAKE_ADDR2LINE     ${TOOLCHAIN_PREFIX}-addr2line  CACHE INTERNAL "")
set(CMAKE_RANLIB        ${TOOLCHAIN_PREFIX}-ranlib  CACHE INTERNAL "")
set(CMAKE_MAKE_PROGRAM  ${CMAKE_MAKE_PROGRAM} CACHE INTERNAL "Ninja generation program")

set(CMAKE_FIND_ROOT_PATH /usr/bin)

set(CMAKE_C_FLAGS_INIT   "-std=gnu11 -fdata-sections -ffunction-sections -Wno-main -m64" CACHE INTERNAL "c compiler flags")
set(CMAKE_CXX_FLAGS_INIT "-fdata-sections -ffunction-sections -Wno-main -m64" CACHE INTERNAL "cxx compiler flags")
set(CMAKE_ASM_FLAGS_INIT "" CACHE INTERNAL "asm compiler flags")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--gc-sections -static-libgcc -static-libstdc++ -pthread -m64" CACHE INTERNAL "exe link flags")

SET(CMAKE_C_FLAGS_DEBUG "-O0 -g -ggdb3 -fno-inline-small-functions" CACHE INTERNAL "c debug compiler flags")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb3 -fno-inline-small-functions" CACHE INTERNAL "cxx debug compiler flags")
SET(CMAKE_ASM_FLAGS_DEBUG "-g -ggdb3" CACHE INTERNAL "asm debug compiler flags")

SET(CMAKE_C_FLAGS_RELEASE "-O3  -DNDEBUG" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE INTERNAL "cxx release compiler flags")

SET(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb3" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb3" CACHE INTERNAL "cxx release compiler flags")

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


######################################################################
# mltk_toolchain_add_exe_targets
#
# Add additional targets to an executable build target
#
# target - The executabe CMake build target
macro(mltk_toolchain_add_exe_targets target)
  set(_output_dir ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
  set(_output_path ${_output_dir}/${target})

  target_link_options(${target}
  PUBLIC 
    -Wl,-Map,${_output_path}.map
  )

  mltk_load_python()
  mltk_get(MLTK_PLATFORM_NAME)

  add_custom_command(TARGET ${target}
    POST_BUILD 
    COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_DIR}/tools/utils/update_launch_json.py --name ${target} --path \"${_output_path}\" --platform ${MLTK_PLATFORM_NAME} --workspace \"${CMAKE_SOURCE_DIR}\"
    COMMAND ${CMAKE_SIZE} ${_output_path}
    COMMENT "Application ${target} binary size"
  )

endmacro()