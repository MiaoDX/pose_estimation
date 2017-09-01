project(relative-pose-5-pint)

cmake_minimum_required(VERSION 2.8)

#include( $ENV{OpenCV2_DIR}/OpenCVConfig.cmake )
#include( $ENV{OpenCV3_DIR}/OpenCVConfig.cmake )
include( $ENV{OpenCV33_DIR}/OpenCVConfig.cmake )
add_definitions(-D_CV_VERSION_3) # `#ifdef _CV_VERSION_3`

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

set(FIVE_POINT_DIR ${CMAKE_SOURCE_DIR}/5point)
set(GET_RT_ALGO ${CMAKE_SOURCE_DIR}/getRTAlgo)
# set(POSE_ESTIMATION_2D2D ${CMAKE_SOURCE_DIR}/pose_estimation_2d2d)
set(POSE_ESTIMATION ${CMAKE_SOURCE_DIR}/pose_estimation)

include_directories( ${FIVE_POINT_DIR} ${GET_RT_ALGO} ${POSE_ESTIMATION} ${OpenCV_INCLUDE_DIRS} )

file(GLOB FIVE_POINT_CPP_FILES ${FIVE_POINT_DIR}/5point.cpp ${FIVE_POINT_DIR}/Rpoly.cpp)
file(GLOB getRTAlgo_FILE ${GET_RT_ALGO}/getRTAlgo.cpp)
# AUX_SOURCE_DIRECTORY( ${POSE_ESTIMATION} POSE_ESTIMATION_FILES) # or below
file(GLOB POSE_ESTIMATION_FILES ${POSE_ESTIMATION}/keypoints_descriptors_utils.cpp ${POSE_ESTIMATION}/refine_matches.cpp ${POSE_ESTIMATION}/pose_estimation_utils.cpp ${POSE_ESTIMATION}/image_utils.cpp)

set(ALL_NECESSARY_FILES ${getRTAlgo_FILE} ${FIVE_POINT_CPP_FILES} ${POSE_ESTIMATION_FILES})


add_executable(perfectPoints perfectPoints.cpp ${ALL_NECESSARY_FILES})
target_link_libraries(perfectPoints ${OpenCV_LIBS})

add_executable(five_point_main ${FIVE_POINT_DIR}/main.cpp ${FIVE_POINT_CPP_FILES})
target_link_libraries(five_point_main ${OpenCV_LIBS})

add_executable(pose_estimation_test pose_estimation_test.cpp ${ALL_NECESSARY_FILES})
target_link_libraries(pose_estimation_test ${OpenCV_LIBS})