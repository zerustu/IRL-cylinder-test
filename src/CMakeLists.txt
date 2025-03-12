# cylinder_test/CMakeLists.txt
set(CURRENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# Define the executable target
target_sources(amr_generation PRIVATE ${CURRENT_DIR}/amr_generation.cpp)
target_sources(amr_generation PRIVATE ${CURRENT_DIR}/geometry.h)

target_sources(cylinder_verification PRIVATE ${CURRENT_DIR}/cylinder_confirmation.cpp)
target_sources(cylinder_verification PRIVATE ${CURRENT_DIR}/geometry.h)

target_sources(test_configuration PRIVATE ${CURRENT_DIR}/test_config.cpp)
target_sources(test_configuration PRIVATE ${CURRENT_DIR}/geometry.h)

target_sources(cylinder_timing PRIVATE ${CURRENT_DIR}/irl_timing.cpp)
target_sources(cylinder_timing PRIVATE ${CURRENT_DIR}/geometry.h)