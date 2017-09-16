###
# Relative pose estimation
###
include_directories(${CMAKE_SOURCE_DIR}/SfM/Rt_transform)
ADD_EXECUTABLE(relativePosePair relativePosePair.cpp ${CMAKE_SOURCE_DIR}/SfM/Rt_transform/Rt_transform.cpp)
TARGET_LINK_LIBRARIES(relativePosePair
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
