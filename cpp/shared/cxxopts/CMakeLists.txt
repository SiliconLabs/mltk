project(mltk_cxxopts
        VERSION 1.0.0
        DESCRIPTION "MLTK cxxopts project"
        HOMEPAGE_URL "https://github.com/jarro2783/cxxopts"
)
export (PACKAGE ${PROJECT_NAME})

CPMAddPackage(
NAME cxxopts 
GITHUB_REPOSITORY jarro2783/cxxopts
VERSION 2.2.1
CACHE_VERSION v2.2.1
OPTIONS 
    "CXXOPTS_BUILD_EXAMPLES OFF"
    "CXXOPTS_BUILD_TESTS OFF"
    "CXXOPTS_ENABLE_INSTALL OFF"
)



add_library(${PROJECT_NAME} INTERFACE)
add_library(mltk::cxxopts ALIAS ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} 
INTERFACE 
    cxxopts
)