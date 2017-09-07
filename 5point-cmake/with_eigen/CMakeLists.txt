# Add CMake to [Nghia Ho's 5 points algorithm](http://nghiaho.com/?p=1675)

project(fivePointEigen_project)

# HARD CODED, or specify in the cmake-gui
# set (EIGEN_INCLUDE_DIR C:/tools/sfm_necessary/eigen_334)




cmake_minimum_required(VERSION 2.8)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)
set(fivePointEigen_version 0.1.5)



# Eigen.
# THE .cmake is copied from [ceres](https://github.com/ceres-solver/ceres-solver/blob/1.13.0/cmake/FindEigen.cmake)
list( APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake )
find_package(Eigen REQUIRED)
message("-- Found Eigen version ${EIGEN_VERSION}: ${EIGEN_INCLUDE_DIRS}")

include_directories(${EIGEN_INCLUDE_DIRS})



include(GenerateExportHeader) # for Cmake functions

file(GLOB FIVE_POINT_SRC_FILES ${CMAKE_SOURCE_DIR}/5point_Eigen/5point.cpp ${CMAKE_SOURCE_DIR}/5point_Eigen/Rpoly.cpp) # not include the main.cpp
file (GLOB FIVE_POINT_SRC_HEADERS ${CMAKE_SOURCE_DIR}/5point_Eigen/*.h)

set (FIVE_POINT_FILES ${FIVE_POINT_SRC_FILES} ${FIVE_POINT_SRC_HEADERS})

message(STATUS "FIVE_POINT_SRC_FILES:${FIVE_POINT_SRC_FILES}")
message(STATUS "FIVE_POINT_SRC_HEADERS:${FIVE_POINT_SRC_HEADERS}")
message(STATUS "FIVE_POINT_FILES:${FIVE_POINT_FILES}")

#User options
set(BUILD_SHARED_LIBS false CACHE BOOL "Build shared library.")

set (target_name fivePointEigen)

# Add the library.
if( BUILD_SHARED_LIBS )
message(STATUS "BUILD_SHARED_LIBS")
## BELOW, creat a full static library -.-
#message(STATUS "BUILD_SHARED_LIBS with TRICK")
#add_library( TMPSTATIC STATIC ${FIVE_POINT_FILES} )
#set_target_properties(TMPSTATIC PROPERTIES DEBUG_POSTFIX -d OUTPUT_NAME ${target_name})
#set_target_properties(TMPSTATIC PROPERTIES OUTPUT_NAME ${target_name})
## TRICK ABOVE


add_library( ${target_name} SHARED ${FIVE_POINT_FILES} )
set_target_properties( ${target_name} PROPERTIES DEBUG_POSTFIX -d )



else()
message(STATUS "BUILD_STATIC_LIBS")
add_library( ${target_name} STATIC ${FIVE_POINT_FILES})
set_target_properties( ${target_name} PROPERTIES DEBUG_POSTFIX -s-d )
set_target_properties( ${target_name} PROPERTIES RELEASE_POSTFIX -s )
endif()

if( WIN32 )
set( SHARE_PATH "." )
set( LIB_PATH "lib" )

elseif(APPLE)
set( SHARE_PATH "${CMAKE_INSTALL_PREFIX}/share/fivePointEigen" )
set( LIB_PATH "lib" )
else()
set( SHARE_PATH "${CMAKE_INSTALL_PREFIX}/share/fivePointEigen" )

if( LIB_SUFFIX )
    set( LIB_PATH "lib${LIB_SUFFIX}" )
else()
    set( LIB_PATH "lib" )
endif()
endif()


generate_export_header(fivePointEigen)
set_property(TARGET fivePointEigen PROPERTY VERSION ${fivePointEigen_version})

install(TARGETS fivePointEigen EXPORT fivePointEigenTargets
LIBRARY DESTINATION lib
ARCHIVE DESTINATION lib
RUNTIME DESTINATION bin
INCLUDES DESTINATION include
)

install(
    FILES
      ${FIVE_POINT_SRC_HEADERS}
    DESTINATION
      include
    COMPONENT
      Devel
  )


include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/fivePointEigen/fivePointEigenVersion.cmake"
    VERSION ${fivePointEigen_version}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT fivePointEigenTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/fivePointEigen/fivePointEigenTargets.cmake"
    NAMESPACE fivePointEigen::
)
configure_file(cmake/fivePointEigenConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/fivePointEigen/fivePointEigenConfig.cmake"
    COPYONLY # not COPY_ONLY
)

set(ConfigPackageLocation lib/cmake/fivePointEigen)
install(EXPORT fivePointEigenTargets
FILE
    fivePointEigenTargets.cmake
NAMESPACE
    fivePointEigen::
DESTINATION
    ${ConfigPackageLocation}
)

install(
FILES
    cmake/fivePointEigenConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/fivePointEigen/fivePointEigenConfig.cmake"
DESTINATION
    ${ConfigPackageLocation}
COMPONENT
    Devel
)


#=============================================================================#
#=============================================================================#


include_directories( ${CMAKE_SOURCE_DIR}/5point_Eigen)
add_executable(example ${CMAKE_SOURCE_DIR}/5point_Eigen/main.cpp)
target_link_libraries(example fivePointEigen)
