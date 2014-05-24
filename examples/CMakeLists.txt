# Minimum version of CMake required. Don't touch.
cmake_minimum_required(VERSION 2.8)

# Set the name of your project
project(mc64_examples)

# Include macros from the SBEL utils library
include(SBELUtils.cmake)

##########################################################
# If using Makefiles, be sure to set CMAKE_BUILD_TYPE to # 
# select different build configurations.                 #
# Possible values:                                       #
# - RELEASE - fully optimized for speed                  #
# - DEBUG - unoptimized, with debugging symbols          #
# - RELWITHDEBUGINFO - optimized, with debugging symbols #
# - MINSIZEREL - optimized for size                      #
# - [not set] - unoptimized                              #
##########################################################


#enable_mpi_support()
#enable_openmp_support()
enable_cuda_support()

# CUSP dependency. Note that we always include CUSP headers using 
# include <cusp/header.h>.
#find_path(CUSP_INCLUDE_DIR 
#          NAMES cusp/array1d.h 
#HINTS ${CUDA_TOOLKIT_INCLUDE} 
#          DOC "Top-level CUSP directory")
	  #if(CUSP_INCLUDE_DIR)
	#message(STATUS "CUSP headers found in ${CUSP_INCLUDE_DIR}/cusp")
   #else()
	#message(FATAL_ERROR "CUSP headers not found. Please set CUSP_INCLUDE_DIR.")
   #endif()
#cuda_include_directories(${CUSP_INCLUDE_DIR})


include_directories(.)
include_directories(..)


SET(MC64_HEADERS
	../../mc64/mc64.h
	../../mc64/common.h
	../../mc64/timer.h
	../../mc64/exception.h
)

SET(MC64_CUHEADERS
	../../mc64/device/mc64.cuh
)

ADD_SUBDIRECTORY(new_mc64)
ADD_SUBDIRECTORY(old_mc64)
