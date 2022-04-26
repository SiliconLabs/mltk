project(mltk_commander
        VERSION 1.0.0
        DESCRIPTION "Silab's Commander Utility"
)
export(PACKAGE ${PROJECT_NAME})


macro(mltk_commander_add_download_target target)
  mltk_load_python()
  mltk_get(MLTK_PLATFORM_NAME)
  mltk_get(MLTK_JLINK_DEVICE)

  if(MLTK_JLINK_DEVICE)
    set(device_arg --device ${MLTK_JLINK_DEVICE})
  endif()

  add_custom_target(${target}_download
    COMMAND ${PYTHON_EXECUTABLE} "${MLTK_DIR}/utils/commander/commander.py" --platform ${MLTK_PLATFORM_NAME} --path "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target}.s37" --masserase ${device_arg}
    DEPENDS ${target}
    COMMENT "Downloading ${target}.s37 to ${MLTK_PLATFORM_NAME}'s flash ..."
    USES_TERMINAL
  )

endmacro()


macro(mltk_commander_add_reset_target target)
  mltk_load_python()
  mltk_get(MLTK_PLATFORM_NAME)
  mltk_get(MLTK_JLINK_DEVICE)

  if(MLTK_JLINK_DEVICE)
    set(device_arg --device ${MLTK_JLINK_DEVICE})
  endif()

  add_custom_target(${target}_reset
    COMMAND ${PYTHON_EXECUTABLE} "${MLTK_DIR}/utils/commander/commander.py" --platform ${MLTK_PLATFORM_NAME} --reset ${device_arg}
    COMMENT "Resetting ${MLTK_PLATFORM_NAME} ..."
    USES_TERMINAL
  )

endmacro()