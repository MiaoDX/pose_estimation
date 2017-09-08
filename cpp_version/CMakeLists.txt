project(relative-pose-5-pint)

cmake_minimum_required(VERSION 2.8)

# GLOBAL setting
set (NECESSARY_LIBS_DIR C:/tools/cmake_install_libs/pose_estimation)

# User options
# User options
set( USE_CV3 FALSE )
set( USE_CV3 TRUE ) # comment this to use OpenCV 2.x

message(STATUS "USE_CV3 value:${USE_CV3}")
if (USE_CV3)
message(STATUS "Using OpenCV 3.x ...")
include( $ENV{OpenCV3_DIR}/OpenCVConfig.cmake )
add_definitions(-D_CV_VERSION_3) # `#ifdef _CV_VERSION_3`

# five point
set (fivePointLibDIR ${NECESSARY_LIBS_DIR}/fivePoint/cv330)
else()
message(STATUS "Using OpenCV 2.x ...")
include( $ENV{OpenCV2_DIR}/OpenCVConfig.cmake )

set (fivePointLibDIR ${NECESSARY_LIBS_DIR}/fivePoint/cv2413)
endif() # (USE_CV3)

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")


set(GET_RT_ALGO ${CMAKE_SOURCE_DIR}/getRTAlgo)
set(POSE_ESTIMATION ${CMAKE_SOURCE_DIR}/pose_estimation)

# five points
message(STATUS "fivePointLibDIR:${fivePointLibDIR}")
include (${fivePointLibDIR}/lib/cmake/fivePoint/fivePointConfig.cmake)
include_directories( ${fivePointLibDIR}/include )


# cluster with alglib
set (AlglibClusterLibDIR ${NECESSARY_LIBS_DIR}/AlglibCluster)
include (${AlglibClusterLibDIR}/lib/cmake/AlglibCluster/AlglibClusterConfig.cmake)
include_directories( ${AlglibClusterLibDIR}/include )

include_directories(${GET_RT_ALGO} ${POSE_ESTIMATION} ${OpenCV_INCLUDE_DIRS} )

file(GLOB getRTAlgo_FILE ${GET_RT_ALGO}/getRTAlgo.cpp)
# AUX_SOURCE_DIRECTORY( ${POSE_ESTIMATION} POSE_ESTIMATION_FILES) # or below
file(GLOB POSE_ESTIMATION_FILES ${POSE_ESTIMATION}/keypoints_descriptors_utils.cpp ${POSE_ESTIMATION}/refine_matches.cpp ${POSE_ESTIMATION}/pose_estimation_utils.cpp ${POSE_ESTIMATION}/image_utils.cpp ${POSE_ESTIMATION}/Rt_transform.cpp)

set(ALL_NECESSARY_FILES ${getRTAlgo_FILE} ${POSE_ESTIMATION_FILES})


set(LINK_LIBS AlglibCluster::AlglibCluster fivePoint::fivePoint ${OpenCV_LIBS})

add_executable(perfectPoints perfectPoints.cpp ${ALL_NECESSARY_FILES})
target_link_libraries(perfectPoints ${LINK_LIBS})

add_executable(pose_estimation_test pose_estimation_test.cpp ${ALL_NECESSARY_FILES})
target_link_libraries(pose_estimation_test ${LINK_LIBS})

add_executable(Rt_transform_test ${POSE_ESTIMATION}/Rt_transform_test.cpp ${POSE_ESTIMATION}/Rt_transform.cpp)
target_link_libraries(Rt_transform_test ${LINK_LIBS})