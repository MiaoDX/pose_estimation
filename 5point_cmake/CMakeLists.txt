project(relative-pose-5-pint)

cmake_minimum_required(VERSION 2.8)

#include( $ENV{OpenCV2_DIR}/OpenCVConfig.cmake )
include( $ENV{OpenCV3_DIR}/OpenCVConfig.cmake )
add_definitions(-D_CV_VERSION_3) # `#ifdef _CV_VERSION_3` in timeutil.h

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

set(FIVE_POINT_DIR ${CMAKE_SOURCE_DIR}/5point)
set(GET_RT_ALGO ${CMAKE_SOURCE_DIR}/getRTAlgo)

include_directories( ${FIVE_POINT_DIR} ${GET_RT_ALGO} ${OpenCV_INCLUDE_DIRS} )

# AUX_SOURCE_DIRECTORY( ${FIVE_POINT_DIR} OTHER_SOURCE_FILES)
file(GLOB FIVE_POINT_CPP_FILES ${FIVE_POINT_DIR}/5point.cpp ${FIVE_POINT_DIR}/Rpoly.cpp ${GET_RT_ALGO}/getRTAlgo.cpp )

 add_executable(demo demo.cpp ${FIVE_POINT_CPP_FILES})
 target_link_libraries(demo ${OpenCV_LIBS})

add_executable(main ${FIVE_POINT_DIR}/main.cpp ${FIVE_POINT_CPP_FILES})
# add_executable(main ${FIVE_POINT_DIR}/main.cpp)
target_link_libraries(main ${OpenCV_LIBS})

