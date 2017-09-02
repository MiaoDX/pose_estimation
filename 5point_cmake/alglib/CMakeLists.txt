project(alglib_test)

cmake_minimum_required(VERSION 2.8)

include( $ENV{OpenCV33_DIR}/OpenCVConfig.cmake )

set(ALGLIB_DIR ${CMAKE_SOURCE_DIR}/src)

include_directories( ${ALGLIB_DIR} ${CMAKE_SOURCE_DIR} )

AUX_SOURCE_DIRECTORY( ${ALGLIB_DIR} ALGLIB_FILES)

add_executable(example_clst_kclusters example_clst_kclusters.cpp ${ALGLIB_FILES})

add_executable(refine_with_clst_kclusters_test refine_with_clst_kclusters_test.cpp refine_with_clst_kclusters.cpp ${ALGLIB_FILES})
target_link_libraries(refine_with_clst_kclusters_test ${OpenCV_LIBS})


add_executable(test test.cpp)