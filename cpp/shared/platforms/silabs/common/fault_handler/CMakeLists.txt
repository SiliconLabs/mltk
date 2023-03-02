
add_library(mltk_platform_fault_handler)
add_library(mltk::platform::fault_handler ALIAS mltk_platform_fault_handler)

target_include_directories(mltk_platform_fault_handler
PUBLIC
  .
)
target_sources(mltk_platform_fault_handler
PRIVATE
  mltk_fault_handler.c
)


target_link_libraries(mltk_platform_fault_handler
PRIVATE
  ${MLTK_PLATFORM}
)


target_link_options(mltk_platform_fault_handler
PUBLIC
  -uNMI_Handler
  -uHardFault_Handler
  -uDefault_Handler
  -uMemManage_Handler
  -uBusFault_Handler
  -uUsageFault_Handler
  -uSMU_NS_PRIVILEGED_IRQHandler
  -uSMU_S_PRIVILEGED_IRQHandler
  -uSMU_SECURE_IRQHandler
)