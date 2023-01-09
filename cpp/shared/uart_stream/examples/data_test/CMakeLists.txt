

# To run this example, add the following to the file:
# <mltk repo root>/user_options.cmake:
#
# mltk_set(MLTK_TARGET mltk_uart_stream_data_test)
#
# And invoke the CMake target:
# mltk_uart_stream_data_test_download
#
# This will download the firmware to the dev board
# Then, run the corresponding data_test.py script


add_executable(mltk_uart_stream_data_test
    main.cc
)  
mltk_add_exe_targets(mltk_uart_stream_data_test)

target_link_libraries(mltk_uart_stream_data_test
PRIVATE 
    ${MLTK_PLATFORM}
    mltk::uart_stream
)
