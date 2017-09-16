
add_executable(registration_faked_gps_position registration_faked_gps_position.cpp)
target_link_libraries(registration_faked_gps_position
  openMVG_system
  openMVG_features
  openMVG_sfm
  easyexif
  stlplus)

set_property(TARGET registration_faked_gps_position PROPERTY FOLDER OpenMVG/software)
install(TARGETS registration_faked_gps_position DESTINATION bin/)
