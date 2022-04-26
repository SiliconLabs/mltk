
set(NAME mltk_sl_mic)
add_library(${NAME})
add_library(mltk::sl_mic ALIAS ${NAME})

target_include_directories(${NAME}
PUBLIC
  .
)
target_sources(${NAME}
PRIVATE
  mltk_sl_mic_i2s.c
)


target_link_libraries(${NAME}
PRIVATE
  ${MLTK_PLATFORM}
  mltk::cpputils
  mltk::gecko_sdk::dmadrv
  mltk::gecko_sdk::includes
)
