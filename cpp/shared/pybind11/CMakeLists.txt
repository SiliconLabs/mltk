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