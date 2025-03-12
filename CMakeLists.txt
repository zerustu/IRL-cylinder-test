cmake_minimum_required(VERSION 3.9)
project(CYLINDER_TEST VERSION 0.1
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules"
  "${PROJECT_SOURCE_DIR}/cmake/Utils")

file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)

if(EXISTS "${LOC_PATH}")
  message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory.")
endif()

set(SOURCE "${PROJECT_SOURCE_DIR}/src")

find_package(Eigen3) # Provide -D EIGEN_PATH=/path/to/Eigen
find_package(CGAL)
find_package(OpenMP REQUIRED)

# Add AMR executable
add_executable(amr_generation)
set_target_properties(amr_generation
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  COMPILE_FLAGS "${CMAKE_CXX_FLAGS}"
  INTERPROCEDURAL_OPTIMIZATION False
  CXX_STANDARD 17)

target_include_directories(amr_generation PRIVATE "./")
target_link_libraries(amr_generation PUBLIC "${IRL_INSTALL_LOCATION}/lib/libirl.a")
target_link_libraries(amr_generation PUBLIC "${IRL_INSTALL_LOCATION}/lib/libabsl_all.a")
target_link_libraries(amr_generation PUBLIC "-lpthread")
target_link_libraries(amr_generation PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(amr_generation PUBLIC "-lquadmath")

if(APPLE)
  target_link_libraries(amr_generation PUBLIC "-framework CoreFoundation")
endif()

target_include_directories(amr_generation PUBLIC "${IRL_INSTALL_LOCATION}/include")
target_include_directories(amr_generation PUBLIC "${IRL_ROOT_LOCATION}")
target_include_directories(amr_generation SYSTEM PUBLIC "${IRL_ROOT_LOCATION}/external/abseil-cpp/")
target_include_directories(amr_generation SYSTEM PUBLIC "${IRL_ROOT_LOCATION}/external/triangle/")
target_include_directories(amr_generation SYSTEM PUBLIC "${JSON}")
target_link_libraries(amr_generation INTERFACE PUBLIC Eigen3::Eigen)
target_link_libraries(amr_generation INTERFACE PUBLIC CGAL)
target_include_directories(amr_generation SYSTEM PUBLIC "")

# Add cylinder_verification executable
add_executable(cylinder_verification)
set_target_properties(cylinder_verification
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  COMPILE_FLAGS "${CMAKE_CXX_FLAGS}"
  INTERPROCEDURAL_OPTIMIZATION False
  CXX_STANDARD 17)

target_include_directories(cylinder_verification PRIVATE "./")
target_link_libraries(cylinder_verification PUBLIC "${IRL_INSTALL_LOCATION}/lib/libirl.a")
target_link_libraries(cylinder_verification PUBLIC "${IRL_INSTALL_LOCATION}/lib/libabsl_all.a")
target_link_libraries(cylinder_verification PUBLIC "-lpthread")
target_link_libraries(cylinder_verification PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(cylinder_verification PUBLIC "-lquadmath")

if(APPLE)
  target_link_libraries(cylinder_verification PUBLIC "-framework CoreFoundation")
endif()

target_include_directories(cylinder_verification PUBLIC "${IRL_INSTALL_LOCATION}/include")
target_include_directories(cylinder_verification PUBLIC "${IRL_ROOT_LOCATION}")
target_include_directories(cylinder_verification SYSTEM PUBLIC "${IRL_ROOT_LOCATION}/external/abseil-cpp/")
target_include_directories(cylinder_verification SYSTEM PUBLIC "${IRL_ROOT_LOCATION}/external/triangle/")
target_include_directories(cylinder_verification SYSTEM PUBLIC "${JSON}")
target_link_libraries(cylinder_verification INTERFACE PUBLIC Eigen3::Eigen)
target_link_libraries(cylinder_verification INTERFACE PUBLIC CGAL)
target_include_directories(cylinder_verification SYSTEM PUBLIC "")

# Add test_configuration executable
add_executable(test_configuration)
set_target_properties(test_configuration
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  COMPILE_FLAGS "${CMAKE_CXX_FLAGS}"
  INTERPROCEDURAL_OPTIMIZATION False
  CXX_STANDARD 17)

target_include_directories(test_configuration PRIVATE "./")
target_link_libraries(test_configuration PUBLIC "${IRL_INSTALL_LOCATION}/lib/libirl.a")
target_link_libraries(test_configuration PUBLIC "${IRL_INSTALL_LOCATION}/lib/libabsl_all.a")
target_link_libraries(test_configuration PUBLIC "-lpthread")
target_link_libraries(test_configuration PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(test_configuration PUBLIC "-lquadmath")

if(APPLE)
  target_link_libraries(test_configuration PUBLIC "-framework CoreFoundation")
endif()

target_include_directories(test_configuration PUBLIC "${IRL_INSTALL_LOCATION}/include")
target_include_directories(test_configuration PUBLIC "${IRL_ROOT_LOCATION}")
target_include_directories(test_configuration SYSTEM PUBLIC "${IRL_ROOT_LOCATION}/external/abseil-cpp/")
target_include_directories(test_configuration SYSTEM PUBLIC "${IRL_ROOT_LOCATION}/external/triangle/")
target_include_directories(test_configuration SYSTEM PUBLIC "${JSON}")
target_link_libraries(test_configuration INTERFACE PUBLIC Eigen3::Eigen)
target_link_libraries(test_configuration INTERFACE PUBLIC CGAL)
target_include_directories(test_configuration SYSTEM PUBLIC "")

# Add timing executable
add_executable(cylinder_timing)
set_target_properties(cylinder_timing
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
  COMPILE_FLAGS "${CMAKE_CXX_FLAGS}"
  INTERPROCEDURAL_OPTIMIZATION False
  CXX_STANDARD 17)

target_include_directories(cylinder_timing PRIVATE "./")
target_link_libraries(cylinder_timing PUBLIC "${IRL_INSTALL_LOCATION}/lib/libirl.a")
target_link_libraries(cylinder_timing PUBLIC "${IRL_INSTALL_LOCATION}/lib/libabsl_all.a")
target_link_libraries(cylinder_timing PUBLIC "-lpthread")
target_link_libraries(cylinder_timing PUBLIC OpenMP::OpenMP_CXX)

target_link_libraries(cylinder_timing PUBLIC "-lquadmath")

if(APPLE)
  target_link_libraries(cylinder_timing PUBLIC "-framework CoreFoundation")
endif()

target_include_directories(cylinder_timing PUBLIC "${IRL_INSTALL_LOCATION}/include")
target_include_directories(cylinder_timing PUBLIC "${IRL_ROOT_LOCATION}")
target_include_directories(cylinder_timing SYSTEM PUBLIC "${IRL_ROOT_LOCATION}/external/abseil-cpp/")
target_include_directories(cylinder_timing SYSTEM PUBLIC "${IRL_ROOT_LOCATION}/external/triangle/")
target_include_directories(cylinder_timing SYSTEM PUBLIC "${JSON}")
target_link_libraries(cylinder_timing INTERFACE PUBLIC Eigen3::Eigen)
target_link_libraries(cylinder_timing INTERFACE PUBLIC CGAL)
target_include_directories(cylinder_timing SYSTEM PUBLIC "")

add_subdirectory("${SOURCE}")