project(mltk_commander
        VERSION 1.0.0
        DESCRIPTION "Silab's Commander Utility"
)
export(PACKAGE ${PROJECT_NAME})


macro(mltk_commander_add_download_target target)
  mltk_load_python()
  mltk_get(MLTK_PLATFORM_NAME)
  mltk_get(MLTK_JLINK_DEVICE)
  mltk_get(MLTK_JLINK_SERIAL_NUMBER)
  mltk_get(MLTK_DOWNLOAD_RUN_SERIAL_NUMBER)
  mltk_get(MLTK_DOWNLOAD_RUN_IP_ADDRESS)
  mltk_get(MLTK_DOWNLOAD_RUN_SETUP_SCRIPT)
  mltk_get(MLTK_DOWNLOAD_RUN_SETUP_SCRIPT_ARGS)

  if(MLTK_JLINK_DEVICE)
    set(device_arg --device ${MLTK_JLINK_DEVICE})
  endif()


  if(MLTK_DOWNLOAD_RUN_SERIAL_NUMBER)
    set(serial_number_arg --serial-number ${MLTK_DOWNLOAD_RUN_SERIAL_NUMBER})
  elseif(MLTK_JLINK_SERIAL_NUMBER)
    set(serial_number_arg --serial-number ${MLTK_JLINK_SERIAL_NUMBER})
  endif()

  if(MLTK_DOWNLOAD_RUN_IP_ADDRESS)
    set(ip_address_arg --ip-address ${MLTK_DOWNLOAD_RUN_IP_ADDRESS})
  endif()

  if(MLTK_DOWNLOAD_RUN_SETUP_SCRIPT)
    if(MLTK_DOWNLOAD_RUN_SETUP_SCRIPT_ARGS)
      separate_arguments(_setup_script_args NATIVE_COMMAND ${MLTK_DOWNLOAD_RUN_SETUP_SCRIPT_ARGS})
    endif()
    add_custom_target(${target}_run_setup_script
      COMMAND ${PYTHON_EXECUTABLE} "${MLTK_DOWNLOAD_RUN_SETUP_SCRIPT}" ${_setup_script_args}
      DEPENDS ${target}
      COMMENT "Executing ${PYTHON_EXECUTABLE} ${MLTK_DOWNLOAD_RUN_SETUP_SCRIPT} ${MLTK_DOWNLOAD_RUN_SETUP_SCRIPT_ARGS}"
      USES_TERMINAL
    )
  else()
    add_custom_target(  ${target}_run_setup_script )
  endif()

  add_custom_target(${target}_download
    COMMAND ${PYTHON_EXECUTABLE} "${MLTK_DIR}/utils/commander/commander.py" --platform ${MLTK_PLATFORM_NAME} --path "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}.s37" --clear_end_of_flash ${device_arg} ${serial_number_arg} ${ip_address_arg}
    DEPENDS ${target} ${target}_run_setup_script
    COMMENT "Downloading ${target}.s37 to ${MLTK_PLATFORM_NAME}'s flash ..."
    USES_TERMINAL
  )

endmacro()


macro(mltk_commander_add_reset_target target)
  mltk_load_python()
  mltk_get(MLTK_PLATFORM_NAME)
  mltk_get(MLTK_JLINK_DEVICE)
  mltk_get(MLTK_JLINK_SERIAL_NUMBER)
  mltk_get(MLTK_DOWNLOAD_RUN_SERIAL_NUMBER)
  mltk_get(MLTK_DOWNLOAD_RUN_IP_ADDRESS)

  if(MLTK_JLINK_DEVICE)
    set(device_arg --device ${MLTK_JLINK_DEVICE})
  endif()

  if(MLTK_DOWNLOAD_RUN_SERIAL_NUMBER)
    set(serial_number_arg --serial-number ${MLTK_DOWNLOAD_RUN_SERIAL_NUMBER})
  elseif(MLTK_JLINK_SERIAL_NUMBER)
    set(serial_number_arg --serial-number ${MLTK_JLINK_SERIAL_NUMBER})
  endif()

  if(MLTK_DOWNLOAD_RUN_IP_ADDRESS)
    set(ip_address_arg --ip-address ${MLTK_DOWNLOAD_RUN_IP_ADDRESS})
  endif()

  add_custom_target(${target}_reset
    COMMAND ${PYTHON_EXECUTABLE} "${MLTK_DIR}/utils/commander/commander.py" --platform ${MLTK_PLATFORM_NAME} --reset ${device_arg} ${serial_number_arg} ${ip_address_arg}
    COMMENT "Resetting ${MLTK_PLATFORM_NAME} ..."
    USES_TERMINAL
  )

endmacro()


###########################################################
# mltk_commander_add_download_run_target
#
# Run the given target's firmware application on an embedded device.
#
# target - CMake target name of the embedded executable
#
# Optional arguments:
# IMAGE <path> - Path to firmware image to download
#                If omitted then use ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}.s37
# TIMEOUT <seconds> - Number of seconds to wait before timing out when using the download_run target
# COMPLETED_MSG <msg> - Console msg app prints to be considered complete
macro(mltk_commander_add_download_run_target target)
  cmake_parse_arguments(DOWNLOAD_RUN_ARG "" "IMAGE;TIMEOUT;COMPLETED_MSG" "" ${ARGN})
  mltk_load_python()
  mltk_get(MLTK_PLATFORM_NAME)
  mltk_get(MLTK_JLINK_DEVICE)
  mltk_get(MLTK_JLINK_SERIAL_NUMBER)
  mltk_get(MLTK_DOWNLOAD_RUN_SETUP_SCRIPT)
  mltk_get(MLTK_DOWNLOAD_RUN_SETUP_SCRIPT_ARGS)
  mltk_get(MLTK_DOWNLOAD_RUN_PROGRAM_SCRIPT)
  mltk_get(MLTK_DOWNLOAD_RUN_PROGRAM_SCRIPT_ARGS)
  mltk_get(MLTK_DOWNLOAD_RUN_RESET_SCRIPT)
  mltk_get(MLTK_DOWNLOAD_RUN_RESET_SCRIPT_ARGS)
  mltk_get(MLTK_DOWNLOAD_RUN_PORT)
  mltk_get(MLTK_DOWNLOAD_RUN_BAUD)
  mltk_get(MLTK_DOWNLOAD_RUN_SERIAL_NUMBER)
  mltk_get(MLTK_DOWNLOAD_RUN_IP_ADDRESS)
  mltk_get(MLTK_DOWNLOAD_RUN_TIMEOUT)
  mltk_get(MLTK_DOWNLOAD_RUN_RETRIES)
  mltk_get(MLTK_DOWNLOAD_RUN_NO_MASSERASE)
  mltk_get(MLTK_DOWNLOAD_RUN_START_MSG)
  mltk_get(MLTK_DOWNLOAD_RUN_COMPLETED_MSG)
  mltk_get(MLTK_DOWNLOAD_RUN_REMOTE_ADDRESS)

  if(MLTK_JLINK_DEVICE)
    set(device_arg --device ${MLTK_JLINK_DEVICE})
  endif()

  if(MLTK_DOWNLOAD_RUN_SETUP_SCRIPT)
    set(setup_script_arg --setup-script "${MLTK_DOWNLOAD_RUN_SETUP_SCRIPT}")
  endif()
  if(MLTK_DOWNLOAD_RUN_SETUP_SCRIPT_ARGS)
    set(setup_script_args_arg --setup-script-args ${MLTK_DOWNLOAD_RUN_SETUP_SCRIPT_ARGS})
  endif()

  if(MLTK_DOWNLOAD_RUN_PROGRAM_SCRIPT)
    set(program_script_arg --program-script "${MLTK_DOWNLOAD_RUN_PROGRAM_SCRIPT}")
  endif()
  if(MLTK_DOWNLOAD_RUN_PROGRAM_SCRIPT_ARGS)
    set(program_script_args_arg --program-script-args ${MLTK_DOWNLOAD_RUN_PROGRAM_SCRIPT_ARGS})
  endif()

  if(MLTK_DOWNLOAD_RUN_RESET_SCRIPT)
    set(reset_script_arg --reset-script "${MLTK_DOWNLOAD_RUN_RESET_SCRIPT}")
  endif()
  if(MLTK_DOWNLOAD_RUN_RESET_SCRIPT_ARGS)
    set(reset_script_args_arg --reset-script-args ${MLTK_DOWNLOAD_RUN_RESET_SCRIPT_ARGS})
  endif()

  if(MLTK_DOWNLOAD_RUN_PORT)
    set(port_arg --port "${MLTK_DOWNLOAD_RUN_PORT}")
  endif()

  if(MLTK_DOWNLOAD_RUN_BAUD)
    set(baud_arg --baud ${MLTK_DOWNLOAD_RUN_BAUD})
  endif()

  if(MLTK_DOWNLOAD_RUN_SERIAL_NUMBER)
    set(serial_number_arg --serial-number "${MLTK_DOWNLOAD_RUN_SERIAL_NUMBER}")
  elseif(MLTK_JLINK_SERIAL_NUMBER)
    set(serial_number_arg --serial-number "${MLTK_JLINK_SERIAL_NUMBER}")
  endif()

  if(MLTK_DOWNLOAD_RUN_IP_ADDRESS)
    set(ip_address_arg --ip-address "${MLTK_DOWNLOAD_RUN_IP_ADDRESS}")
  endif()

  if(DOWNLOAD_RUN_ARG_TIMEOUT)
    set(timeout_arg --timeout ${DOWNLOAD_RUN_ARG_TIMEOUT})
  elseif(MLTK_DOWNLOAD_RUN_TIMEOUT)
    set(timeout_arg --timeout ${MLTK_DOWNLOAD_RUN_TIMEOUT})
  endif()

  if(MLTK_DOWNLOAD_RUN_RETRIES)
    set(retries_arg --retries ${MLTK_DOWNLOAD_RUN_RETRIES})
  endif()

  if(MLTK_DOWNLOAD_RUN_REMOTE_ADDRESS)
    set(remote_address_arg --remote-address "${MLTK_DOWNLOAD_RUN_REMOTE_ADDRESS}")
  endif()

   if(MLTK_DOWNLOAD_RUN_START_MSG)
    set(start_msg_arg --start-msg "${MLTK_DOWNLOAD_RUN_START_MSG}")
  endif()

  if(DOWNLOAD_RUN_ARG_COMPLETED_MSG)
    set(completed_msg_arg --completed-msg "${DOWNLOAD_RUN_ARG_COMPLETED_MSG}")
  elseif(MLTK_DOWNLOAD_RUN_COMPLETED_MSG)
    set(completed_msg_arg --completed-msg "${MLTK_DOWNLOAD_RUN_COMPLETED_MSG}")
  endif()


  if(NOT MLTK_DOWNLOAD_RUN_NO_MASSERASE)
    set(masserase_arg --masserase)
  endif()

  if(NOT DOWNLOAD_RUN_ARG_IMAGE)
    set(DOWNLOAD_RUN_ARG_IMAGE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}.s37")
  endif()

  set(log_path_arg --log-path "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}.log")

  add_custom_target(${target}_download_run
    COMMAND ${PYTHON_EXECUTABLE} -m mltk build download_run "${DOWNLOAD_RUN_ARG_IMAGE}" --platform ${MLTK_PLATFORM_NAME} ${log_path_arg} --dump-log-file ${masserase_arg} ${device_arg} ${setup_script_arg} ${setup_script_args_arg} ${program_script_arg} ${program_script_args_arg} ${reset_script_arg} ${reset_script_args_arg} ${port_arg} ${baud_arg} ${serial_number_arg} ${ip_address_arg} ${timeout_arg} ${retries_arg} ${remote_address_arg} ${start_msg_arg} ${completed_msg_arg}
    DEPENDS ${target}
    COMMENT "Downloading and running ${DOWNLOAD_RUN_ARG_IMAGE} on ${MLTK_PLATFORM_NAME}'s ..."
    USES_TERMINAL
  )

endmacro()