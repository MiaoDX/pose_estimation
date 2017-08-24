
cmake_minimum_required(VERSION 2.8)

project(gpu_testing)

set(EXECUTABLE_OUTPUT_PATH ../bin)

set(LIBRARY_OUTPUT_PATH ../lib)



#find_package(OpenCV REQUIRED)
include( $ENV{OpenCV33_DIR}/OpenCVConfig.cmake )



find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "-arch=sm_30" CACHE STRING "nvcc flags" FORCE)

set(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)



include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

cuda_add_executable(main surf_keypoint_matcher.cpp)

target_link_libraries(main ${CUDA_LIBRARIES} ${OpenCV_LIBS})