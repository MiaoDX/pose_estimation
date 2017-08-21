project(relative-pose-5-pint)

cmake_minimum_required(VERSION 2.8)

include( $ENV{OpenCV2_DIR}/OpenCVConfig.cmake )
#include( $ENV{OpenCV3_DIR}/OpenCVConfig.cmake )
#add_definitions(-D_CV_VERSION_3) # `#ifdef _CV_VERSION_3`

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

set(FIVE_POINT_DIR ${CMAKE_SOURCE_DIR}/5point)
set(GET_RT_ALGO ${CMAKE_SOURCE_DIR}/getRTAlgo)
set(POSE_ESTIMATION_2D2D ${CMAKE_SOURCE_DIR}/pose_estimation_2d2d)

include_directories( ${FIVE_POINT_DIR} ${GET_RT_ALGO} ${POSE_ESTIMATION_2D2D} ${OpenCV_INCLUDE_DIRS} )

# AUX_SOURCE_DIRECTORY( ${FIVE_POINT_DIR} OTHER_SOURCE_FILES)
file(GLOB FIVE_POINT_CPP_FILES ${FIVE_POINT_DIR}/5point.cpp ${FIVE_POINT_DIR}/Rpoly.cpp ${GET_RT_ALGO}/getRTAlgo.cpp)
file(GLOB FIVE_POINT_AND_2D2D_CPP_FILES ${FIVE_POINT_DIR}/5point.cpp ${FIVE_POINT_DIR}/Rpoly.cpp ${GET_RT_ALGO}/getRTAlgo.cpp ${POSE_ESTIMATION_2D2D}/pose_estimation_2d2d.cpp)


add_executable(perfectPoints perfectPoints.cpp ${FIVE_POINT_CPP_FILES})
target_link_libraries(perfectPoints ${OpenCV_LIBS})

add_executable(five_point_main ${FIVE_POINT_DIR}/main.cpp ${FIVE_POINT_CPP_FILES})
target_link_libraries(five_point_main ${OpenCV_LIBS})


add_executable(pose_estimation_2d2d_test pose_estimation_2d2d_test.cpp ${FIVE_POINT_AND_2D2D_CPP_FILES} )
target_link_libraries(pose_estimation_2d2d_test ${OpenCV_LIBS})


add_executable(pose_estimation_test pose_estimation_test.cpp ${FIVE_POINT_AND_2D2D_CPP_FILES})
target_link_libraries(pose_estimation_test ${OpenCV_LIBS})


add_executable(pose_estimation_refine_test pose_estimation_refine_test.cpp ${FIVE_POINT_AND_2D2D_CPP_FILES})
target_link_libraries(pose_estimation_refine_test ${OpenCV_LIBS})