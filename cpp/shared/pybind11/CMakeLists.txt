project(mltk_pybind11
        VERSION 1.0.0
        DESCRIPTION "MLTK Pybind11"
)
export(PACKAGE ${PROJECT_NAME})

mltk_load_python()


# Let pybind re-find python
# The cached variable: PYTHON_EXECUTABLE
# should already be defined so pybind11 should just use that
set(Python_FOUND OFF)
set(Python3_FOUND OFF)
set(Python3_FOUND OFF)


CPMAddPackage(
    NAME pybind11
    GITHUB_REPOSITORY pybind/pybind11
    GIT_TAG v2.6.2
    CACHE_VERSION v2.6.2
    OPTIONS 
        "MSVC OFF"
)


add_library(mltk_pybind11_helper INTERFACE)
add_library(mltk::pybind11_helper ALIAS mltk_pybind11_helper)

target_sources(mltk_pybind11_helper
INTERFACE
  pybind11_helper.hpp
)
target_include_directories(mltk_pybind11_helper
INTERFACE
  .
  ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(mltk_pybind11_helper
INTERFACE 
  pybind11_headers
  ${PYTHON_LIBRARIES}
  mltk_tflite_micro # NOTE: These must be manually included into the build
  mltk_tflite_micro_model
)
