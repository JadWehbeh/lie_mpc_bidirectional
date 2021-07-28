cmake_minimum_required(VERSION 3.10)
project(lie_mpc_bidirectional VERSION 1.0.0 LANGUAGES CXX)
include(GNUInstallDirs)
add_library(${PROJECT_NAME} INTERFACE)
find_package(ALGLIB REQUIRED)
find_library(ALGLIB_LIBRARY NAMES alglib libalglib HINTS /usr/lib)
include_directories(${ALGLIB_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
  $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(${PROJECT_NAME} INTERFACE ${ALGLIB_LIBRARY})

find_package(Catch2 REQUIRED)
add_executable(mpc_test test/test.cpp)
target_link_libraries(mpc_test PRIVATE Catch2::Catch2)
target_link_libraries(mpc_test PRIVATE ${ALGLIB_LIBRARY})
target_include_directories(mpc_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
include(CTest)
include(Catch)
catch_discover_tests(mpc_test)
enable_testing()
