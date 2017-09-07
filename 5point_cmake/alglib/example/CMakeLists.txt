project(AlglibClusterTest)

cmake_minimum_required(VERSION 2.8)

set (AlglibClusterLibDIR C:/tools/cmake_install_libs/pose_estimation/AlglibCluster)

include (${AlglibClusterLibDIR}/lib/cmake/AlglibCluster/AlglibClusterConfig.cmake)

include_directories( ${AlglibClusterLibDIR}/include )

add_executable(example_clst_kclusters example_clst_kclusters.cpp)
target_link_libraries(example_clst_kclusters AlglibCluster::AlglibCluster)

add_executable(refine_with_clst_kclusters_test refine_with_clst_kclusters_test.cpp)
target_link_libraries(refine_with_clst_kclusters_test AlglibCluster::AlglibCluster)