project(fivePointTest)

cmake_minimum_required(VERSION 2.8)


# GLOBAL setting
set (NECESSARY_LIBS_DIR C:/tools/cmake_install_libs/pose_estimation)

# User options
set( USE_CV3 FALSE )
set( USE_CV3 TRUE ) # comment this to use OpenCV 2.x

message(STATUS "USE_CV3 value:${USE_CV3}")
if (USE_CV3)
message(STATUS "Using OpenCV 3.x:")
include( $ENV{OpenCV3_DIR}/OpenCVConfig.cmake )
# five point
set (fivePointLibDIR ${NECESSARY_LIBS_DIR}/fivePoint/cv330)
else()
message(STATUS "Using OpenCV 2.x:")
include( $ENV{OpenCV2_DIR}/OpenCVConfig.cmake )
set (fivePointLibDIR ${NECESSARY_LIBS_DIR}/fivePoint/cv2413)
endif()

message(STATUS "OpenCV_LIBS:${OpenCV_LIBS}")
message(STATUS "fivePointLibDIR:${fivePointLibDIR}")

include (${fivePointLibDIR}/lib/cmake/fivePoint/fivePointConfig.cmake)


include_directories( ${fivePointLibDIR}/include ${OpenCV_INCLUDE_DIRS})

add_executable(example ../5point/main.cpp)
target_link_libraries(example fivePoint::fivePoint ${OpenCV_LIBS}) 