#cuda_include_directories(../)
#cuda_include_directories(../..)

SOURCE_GROUP("MC64 Headers" FILES ${MC64_HEADERS})
SOURCE_GROUP("MC64 CUDA Headers" FILES ${MC64_CUHEADERS})

cuda_add_executable(driver_new_mc64 driver_new_mc64.cu ${SPIKE_HEADERS} ${SPIKE_CUHEADERS})
