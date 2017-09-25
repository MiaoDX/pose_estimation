# Export no symbols by default (if the compiler supports it).
# This makes e.g. GCC's "visibility behavior" consistent with MSVC's. 
# On Windows/MSVC this is a noop.
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# Always include the source and build directories in the include path.
set(CMAKE_INCLUDE_CURRENT_DIR ON)


###
# Relative pose estimation
###
include_directories(${CMAKE_SOURCE_DIR}/SfM)
include_directories(${CMAKE_SOURCE_DIR}/SfM/Rt_transform)

# create the shared library
add_library(relativePosePair SHARED relativePosePair.cpp ${CMAKE_SOURCE_DIR}/SfM/Rt_transform/Rt_transform.cpp)
TARGET_LINK_LIBRARIES(relativePosePair
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  stlplus
  )

# generate export header automatically
include (GenerateExportHeader)
GENERATE_EXPORT_HEADER(relativePosePair
             BASE_NAME relativePosePair
             EXPORT_MACRO_NAME SHARED_EXPORT
             EXPORT_FILE_NAME shared_EXPORTS.h
             STATIC_DEFINE SHARED_EXPORTS_BUILT_AS_STATIC)


ADD_EXECUTABLE(relativePosePair_test relativePosePair_test.cpp)
TARGET_LINK_LIBRARIES(relativePosePair_test
  relativePosePair
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  stlplus
  )

###
# Intrinsic image analysis and SfM_Data container initialization
###
ADD_EXECUTABLE(SfMInit_ImageListing main_SfMInit_ImageListing.cpp)
TARGET_LINK_LIBRARIES(SfMInit_ImageListing
  openMVG_system
  openMVG_image
  openMVG_features
  openMVG_sfm
  easyexif
  )

# Installation rules
SET_PROPERTY(TARGET SfMInit_ImageListing PROPERTY FOLDER OpenMVG/software)
INSTALL(TARGETS SfMInit_ImageListing DESTINATION bin/)



# -
# Use openCV Feature detector to describe the images
# -
IF(OpenMVG_USE_OPENCV)
  ADD_EXECUTABLE(openMVG_main_ComputeFeatures_OpenCV
    main_ComputeFeatures_OpenCV.cpp)
  TARGET_LINK_LIBRARIES(openMVG_main_ComputeFeatures_OpenCV
    openMVG_system
    openMVG_image
    openMVG_features
    openMVG_multiview
    openMVG_sfm
    stlplus
    vlsift
    ${OpenCV_LIBS})
  target_include_directories(openMVG_main_ComputeFeatures_OpenCV PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_compile_definitions(openMVG_main_ComputeFeatures PRIVATE USE_OCVSIFT)

  SET_PROPERTY(TARGET openMVG_main_ComputeFeatures_OpenCV PROPERTY FOLDER OpenMVG/software)
  SET_PROPERTY(TARGET openMVG_main_ComputeFeatures_OpenCV PROPERTY INSTALL_RPATH_USE_LINK_PATH TRUE)
  INSTALL(TARGETS openMVG_main_ComputeFeatures_OpenCV DESTINATION bin/)
ENDIF(OpenMVG_USE_OPENCV)


#add_subdirectory(clustering)
