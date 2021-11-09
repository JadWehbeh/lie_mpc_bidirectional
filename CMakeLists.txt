cmake_minimum_required(VERSION 3.10)
project(lie_mpc_bidirectional VERSION 1.0.0 LANGUAGES CXX)
include(GNUInstallDirs)
add_library(${PROJECT_NAME} INTERFACE)
find_package(ALGLIB REQUIRED)
find_package(Eigen3 REQUIRED)
find_library(ALGLIB_LIBRARY NAMES alglib libalglib HINTS /usr/lib)
include_directories(${ALGLIB_INCLUDE_DIRS})
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_11)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
  $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_include_directories(${PROJECT_NAME} INTERFACE ${ALGLIB_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} INTERFACE ${ALGLIB_LIBRARY})
target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

find_package(Catch2 REQUIRED)
add_executable(mpc_test test/test.cpp)
target_link_libraries(mpc_test PRIVATE Catch2::Catch2)
target_link_libraries(mpc_test PRIVATE ${ALGLIB_LIBRARY})
target_link_libraries(mpc_test PRIVATE Eigen3::Eigen)
target_include_directories(mpc_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
include(CTest)
include(Catch)
catch_discover_tests(mpc_test)
enable_testing()

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}_Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${PROJECT_NAME}ConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
if(NOT INCLUDE_INSTALL_DIR)
  set(INCLUDE_INSTALL_DIR include/lie_mpc_bidirectional)
endif()
configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake
  PATH_VARS INCLUDE_INSTALL_DIR)
install(EXPORT ${PROJECT_NAME}_Targets
  FILE ${PROJECT_NAME}Targets.cmake
  NAMESPACE ${PROJECT_NAME}::
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/cmake)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})
