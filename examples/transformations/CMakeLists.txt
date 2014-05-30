#cuda_include_directories(../)
#cuda_include_directories(../..)

SOURCE_GROUP("MC64 Headers" FILES ${MC64_HEADERS})
SOURCE_GROUP("MC64 CUDA Headers" FILES ${MC64_CUHEADERS})

cuda_add_executable(driver_transformations driver_transformations.cu ${MC64_HEADERS} ${MC64_CUHEADERS})
