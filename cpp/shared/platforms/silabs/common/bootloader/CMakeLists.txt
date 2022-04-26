
add_library(mltk_platform_silabs_simple_bootloader)

target_sources(mltk_platform_silabs_simple_bootloader
PRIVATE 
    simple_bootloader.c
)

target_link_libraries(mltk_platform_silabs_simple_bootloader
PRIVATE 
    ${MLTK_PLATFORM}
)
