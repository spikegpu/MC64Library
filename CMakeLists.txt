CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(mc64)
INCLUDE(cmake/SBELUtils.cmake)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

enable_cuda_support()
MESSAGE(STATUS "Cuda version: ${CUDA_VERSION}")

SET(MC64_HEADERS
  mc64/mc64.h
  mc64/common.h
  mc64/timer.h
  mc64/exception.h
  )

SET(MC64_CUHEADERS
  mc64/device/mc64.cuh
  )

INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
    )

cuda_add_executable(driver driver.cu ${MC64_HEADERS} ${MC64_CUHEADERS})

IF(NOT (${CUDA_VERSION} VERSION_LESS "6.0"))
  SET(MC64_HAS_CUDA6 "#define MC64_HAS_CUDA6")
  cuda_add_executable(driver_um driver_um.cu ${MC64_HEADERS} ${MC64_CUHEADERS})
ENDIF()

CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/cmake/config.in
  ${PROJECT_BINARY_DIR}/mc64/config.h
  )