add_executable (test_decompose test_decompose.cpp)
target_link_libraries(test_decompose ${PROJECT_NAME})

#add_executable (test_operator_norm test_operator_norm.cpp)
#target_link_libraries(test_operator_norm ${PROJECT_NAME})

add_executable (test_decompose_interleave test_decompose_interleave.cpp)
target_link_libraries(test_decompose_interleave ${PROJECT_NAME})

add_executable (test_decompose_interleave_new test_decompose_interleave_new.cpp)
target_link_libraries(test_decompose_interleave_new ${PROJECT_NAME})

add_executable (test_decompose_interleave_hybrid test_decompose_interleave_hybrid.cpp)
target_link_libraries(test_decompose_interleave_hybrid ${PROJECT_NAME})

add_executable (test_decompose_interleave_new_hybrid test_decompose_interleave_new_hybrid.cpp)
target_link_libraries(test_decompose_interleave_new_hybrid ${PROJECT_NAME})

add_executable (test_coeff_decompose_interleave test_coeff_decompose_interleave.cpp)
target_link_libraries(test_coeff_decompose_interleave ${PROJECT_NAME})

add_executable (test_sampling test_sampling.cpp)
target_link_libraries(test_sampling ${PROJECT_NAME})

add_executable (test_coeff_decompose_result test_coeff_decompose_result.cpp)
target_link_libraries(test_coeff_decompose_result ${PROJECT_NAME})