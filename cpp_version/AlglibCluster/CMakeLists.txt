# This is the AlglibCluster lib generator, which is used for determine the constant sublist in a list of vectors
# Used for our `constant ransac` mechanism

# The skeleton is mostly copy from [official example](https://cmake.org/cmake/help/v3.9/manual/cmake-packages.7.html#using-packages)
# The static/shared and debug/release separate method is copied from [Cloaked9000/frnetlib](https://github.com/Cloaked9000/frnetlib/blob/master/CMakeLists.txt)

project(AlglibCluster_project)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)
set(AlglibCluster_version 0.1.0)

include(GenerateExportHeader) # for Cmake functions

set(ALGLIB_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(REFINE_RT_DIR ${CMAKE_SOURCE_DIR}/refine_rt)

AUX_SOURCE_DIRECTORY( ${ALGLIB_SRC_DIR} ALGLIB_SRC_FILES)
file (GLOB ALGLIB_SRC_HEADERS "${ALGLIB_SRC_DIR}/*.h")

set (AlglibCluster_HEADERS ${ALGLIB_SRC_HEADERS} ${REFINE_RT_DIR}/refine_with_clst_kclusters.h)
set (AlglibCluster_FILES ${ALGLIB_SRC_FILES} ${REFINE_RT_DIR}/refine_with_clst_kclusters.cpp)


message(STATUS "AlglibCluster_HEADERS:${AlglibCluster_HEADERS}")
message(STATUS "AlglibCluster_FILES:${AlglibCluster_FILES}")


#User options
set(BUILD_SHARED_LIBS false CACHE BOOL "Build shared library.")

# add_library(AlglibCluster ${AlglibCluster_HEADERS} ${AlglibCluster_FILES} )

# Add the library.
if( BUILD_SHARED_LIBS )
add_library( AlglibCluster SHARED ${AlglibCluster_HEADERS} ${AlglibCluster_FILES} )
set_target_properties( AlglibCluster PROPERTIES DEBUG_POSTFIX -d )
else()
add_library( AlglibCluster ${AlglibCluster_HEADERS} ${AlglibCluster_FILES})
set_target_properties( AlglibCluster PROPERTIES DEBUG_POSTFIX -s-d )
set_target_properties( AlglibCluster PROPERTIES RELEASE_POSTFIX -s )
set_target_properties( AlglibCluster PROPERTIES MINSIZEREL_POSTFIX -s )
endif()

if( WIN32 )
set( SHARE_PATH "." )
set( LIB_PATH "lib" )
elseif(APPLE)
set( SHARE_PATH "${CMAKE_INSTALL_PREFIX}/share/AlglibCluster" )
set( LIB_PATH "lib" )
else()
set( SHARE_PATH "${CMAKE_INSTALL_PREFIX}/share/AlglibCluster" )

if( LIB_SUFFIX )
    set( LIB_PATH "lib${LIB_SUFFIX}" )
else()
    set( LIB_PATH "lib" )
endif()
endif()


generate_export_header(AlglibCluster)
set_property(TARGET AlglibCluster PROPERTY VERSION ${AlglibCluster_version})

install(TARGETS AlglibCluster EXPORT AlglibClusterTargets
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
RUNTIME DESTINATION bin
INCLUDES DESTINATION include
)

install(
    FILES
      ${AlglibCluster_HEADERS}
    DESTINATION
      include
    COMPONENT
      Devel
  )


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/AlglibCluster/AlglibClusterVersion.cmake"
    VERSION ${AlglibCluster_version}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT AlglibClusterTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/AlglibCluster/AlglibClusterTargets.cmake"
    NAMESPACE AlglibCluster::
)
configure_file(cmake/AlglibClusterConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/AlglibCluster/AlglibClusterConfig.cmake"
    COPYONLY # not COPY_ONLY
)

set(ConfigPackageLocation lib/cmake/AlglibCluster)
install(EXPORT AlglibClusterTargets
FILE
    AlglibClusterTargets.cmake
NAMESPACE
    AlglibCluster::
DESTINATION
    ${ConfigPackageLocation}
)

install(
FILES
    cmake/AlglibClusterConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/AlglibCluster/AlglibClusterConfig.cmake"
DESTINATION
    ${ConfigPackageLocation}
COMPONENT
    Devel
)


#=================#
# BELOW ARE TESTS #
#=================#

include_directories( ${ALGLIB_SRC_DIR} ${REFINE_RT_DIR} )


#add_executable(example_clst_kclusters example_clst_kclusters.cpp ${ALGLIB_FILES})
# add_executable(refine_with_clst_kclusters_test refine_with_clst_kclusters_test.cpp refine_with_clst_kclusters.cpp ${ALGLIB_FILES})
# target_link_libraries(refine_with_clst_kclusters_test ${OpenCV_LIBS})

add_executable(example_clst_kclusters example/example_clst_kclusters.cpp)
target_link_libraries(example_clst_kclusters AlglibCluster)

add_executable(refine_with_clst_kclusters_test example/refine_with_clst_kclusters_test.cpp)
target_link_libraries(refine_with_clst_kclusters_test AlglibCluster)