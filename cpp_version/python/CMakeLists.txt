message( STATUS "    In classes cv, link libs are: ${LINK_LIBS}" )

set(LIBNAME RelativePose5Point)
add_library(${LIBNAME} MODULE classes_cv.cpp ${ALL_NECESSARY_FILES})
#target_link_libraries(${LIBNAME} PRIVATE pybind11::module ${OpenCV_LIBS} )
message( STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" )
target_link_libraries(${LIBNAME} PRIVATE pybind11::module ${LINK_LIBS} )


if( MSVC )
set_target_properties(${LIBNAME} PROPERTIES
    SUFFIX ".pyd"
)
else()
set_target_properties(${LIBNAME} PROPERTIES
    SUFFIX ".so"
)
endif()