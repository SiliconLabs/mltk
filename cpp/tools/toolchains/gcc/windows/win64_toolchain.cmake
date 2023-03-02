# Name of the target
set(CMAKE_SYSTEM_NAME Windows)
set(TOOLCHAIN_PREFIX x86_64-w64-mingw32)
set(CMAKE_SYSTEM_PROCESSOR mingw)
set(MLTK_TOOLCHAIN_NAME windows CACHE INTERNAL "")


include(${CMAKE_CURRENT_LIST_DIR}/../../../cmake/utilities.cmake)

string(REGEX REPLACE "/CMakeFiles/CMakeTmp$" "" _bin_dir ${CMAKE_BINARY_DIR})

if(NOT MLTK_USER_OPTIONS)
  set(MLTK_USER_OPTIONS ${CMAKE_SOURCE_DIR}/user_options.cmake)
endif()

if(EXISTS "${MLTK_USER_OPTIONS}" AND NOT MLTK_NO_USER_OPTIONS)
  include("${MLTK_USER_OPTIONS}")
endif()

if(NOT TOOLCHAIN_DIR)
  set(gcc_path_file "${_bin_dir}/gcc_path.txt")
  if(NOT EXISTS ${gcc_path_file})
    mltk_load_python()
    message(NOTICE "Preparing GCC Windows toolchain (this may take awhile) ...")
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
      message(FATAL_ERROR "${_err_msg}\nFailed to download GCC Windows toolchain, see: ${CMAKE_CURRENT_LIST_DIR}/download.log\nAlso see: https://siliconlabs.github.io/mltk/docs/cpp_development/index.html\n\n")
    endif()
    list(GET output 0 TOOLCHAIN_DIR)
    file(WRITE ${gcc_path_file} ${TOOLCHAIN_DIR})
  else()
    file(READ ${gcc_path_file} TOOLCHAIN_DIR)
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
set(TOOLCHAIN_DIR       ${TOOLCHAIN_DIR} CACHE INTERNAL "GCC Win64  toolchain directory")
set(CMAKE_C_COMPILER    ${TOOLCHAIN_DIR}/bin/${TOOLCHAIN_PREFIX}-gcc.exe  CACHE INTERNAL "C compiler")
set(CMAKE_CXX_COMPILER  ${TOOLCHAIN_DIR}/bin/${TOOLCHAIN_PREFIX}-g++.exe  CACHE INTERNAL "C++ compiler")
set(CMAKE_AR            ${TOOLCHAIN_DIR}/bin/${TOOLCHAIN_PREFIX}-gcc-ar.exe  CACHE INTERNAL "AR compiler")
set(CMAKE_AS            ${TOOLCHAIN_DIR}/bin/${TOOLCHAIN_PREFIX}-as.exe  CACHE INTERNAL "ASsembly compiler")
set(CMAKE_NM            ${TOOLCHAIN_DIR}/bin/nm.exe  CACHE INTERNAL "nm")
set(CMAKE_LINKER        ${TOOLCHAIN_DIR}/bin/ld.exe  CACHE INTERNAL "Linker")
set(CMAKE_OBJCOPY       ${TOOLCHAIN_DIR}/bin/objcopy.exe  CACHE INTERNAL "")
set(CMAKE_OBJDUMP       ${TOOLCHAIN_DIR}/bin/objdump.exe  CACHE INTERNAL "")
set(CMAKE_SIZE          ${TOOLCHAIN_DIR}/bin/size.exe  CACHE INTERNAL "")
set(CMAKE_READELF       ${TOOLCHAIN_DIR}/bin/readelf.exe  CACHE INTERNAL "")
set(CMAKE_STRIP         ${TOOLCHAIN_DIR}/bin/strip.exe  CACHE INTERNAL "")
set(CMAKE_ADDR2LINE     ${TOOLCHAIN_DIR}/bin/addr2line.exe  CACHE INTERNAL "")
set(CMAKE_DLLTOOL       ${TOOLCHAIN_DIR}/bin/dlltool.exe  CACHE INTERNAL "")
set(CMAKE_GENDEF        ${TOOLCHAIN_DIR}/bin/gendef.exe  CACHE INTERNAL "")
set(CMAKE_RANLIB        ${TOOLCHAIN_DIR}/bin/ranlib.exe  CACHE INTERNAL "")
set(CMAKE_RC_COMPILER   ${TOOLCHAIN_DIR}/bin/windres.exe  CACHE INTERNAL "")
set(CMAKE_MAKE_PROGRAM  ${CMAKE_MAKE_PROGRAM} CACHE INTERNAL "Ninja generation program")
set(CMAKE_FIND_ROOT_PATH ${TOOLCHAIN_DIR})

set(CMAKE_C_FLAGS_INIT   "-std=gnu11 -fdata-sections -ffunction-sections -m64 -fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=/ -fmacro-prefix-map=${MLTK_CPP_DIR}/=/" CACHE INTERNAL "c compiler flags")
set(CMAKE_CXX_FLAGS_INIT "-fdata-sections -ffunction-sections -m64 -fmacro-prefix-map=${CMAKE_SOURCE_DIR}/=/ -fmacro-prefix-map=${MLTK_CPP_DIR}/=/" CACHE INTERNAL "cxx compiler flags")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,--gc-sections -static-libgcc -static-libstdc++ -pthread -m64 -static" CACHE INTERNAL "exe link flags")

SET(CMAKE_C_FLAGS_DEBUG "-O0 -g -ggdb3 -fno-inline-small-functions" CACHE INTERNAL "c debug compiler flags")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb3 -fno-inline-small-functions" CACHE INTERNAL "cxx debug compiler flags")
SET(CMAKE_ASM_FLAGS_DEBUG "-g -ggdb3" CACHE INTERNAL "asm debug compiler flags")


mltk_get(MLTK_ENABLE_DEBUG_INFO_IN_RELEASE_BUILDS)
if(MLTK_ENABLE_DEBUG_INFO_IN_RELEASE_BUILDS)
SET(CMAKE_C_FLAGS_RELEASE "-O3 -g -ggdb3" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -ggdb3" CACHE INTERNAL "cxx release compiler flags")
else()
SET(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE INTERNAL "cxx release compiler flags")
endif()

SET(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb3" CACHE INTERNAL "c release compiler flags")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -ggdb3" CACHE INTERNAL "cxx release compiler flags")

SET(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1 CACHE INTERNAL "")
SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1 CACHE INTERNAL "")

SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@" CACHE INTERNAL "")
SET(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@" CACHE INTERNAL "")

SET(CMAKE_NINJA_FORCE_RESPONSE_FILE 1 CACHE INTERNAL "")

SET(MSVC OFF CACHE INTERNAL "")


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
    COMMAND ${PYTHON_EXECUTABLE} ${MLTK_CPP_DIR}/tools/utils/update_launch_json.py --name ${target} --path \"${_output_path}\" --toolchain \"${TOOLCHAIN_DIR}/bin\" --platform windows --workspace \"${CMAKE_SOURCE_DIR}\"
    COMMAND ${CMAKE_SIZE} ${_output_path}.exe
    COMMENT "Application ${target} binary size"
  )


  add_custom_target(${target}_download_run
    COMMAND ${_output_path}.exe
    DEPENDS ${_output_path}.exe
    COMMENT "Running ${target}.exe"
    USES_TERMINAL
  )

endmacro()