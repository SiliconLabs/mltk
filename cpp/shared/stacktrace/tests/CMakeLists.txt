add_executable(${PROJECT_NAME}_test)
target_sources(${PROJECT_NAME}_test 
PRIVATE 
    basic_test.c
)


 target_link_libraries(${PROJECT_NAME}_test 
 PRIVATE 
    ${PROJECT_NAME}
 )

# Disabled test as, by design, it causes a segfault which automatically causes the test system to think it failed
# add_test(utility_stacktrace_test ${PROJECT_BINARY_DIR}/utility_stacktrace_test)
# set_tests_properties(utility_stacktrace_test
# 		PROPERTIES
# 		PASS_REGULAR_EXPRESSION "^Error: EXCEPTION_ACCESS_VIOLATION\\s+cause_segfault at")