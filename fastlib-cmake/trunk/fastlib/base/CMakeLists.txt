cmake_minimum_required(VERSION 2.6)

add_subdirectory( config )

#~ include_directories( config
#~                      $ENV{FASTLIBPATH}
#~                      $ENV{FASTLIBPATH}/fastlib
#~                      ${CMAKE_CURRENT_BINARY_DIR}/config
#~                      ${FASTLIB_INCLUDE_DIRS}
#~ )
include_directories( . )

set( TMP_HEADERS 
    ansi_colors.h  cc.h	common.h    debug.h	  fortran.h  otrav_impl.h
    base.h	       ccmem.h	compiler.h  deprecated.h  otrav.h    test.h
)
set( TMP_CXX 
    cc.cc  ccmem.cc  otrav.cc
)
set( TMP_C 
    common.c  debug.c
)
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX} ${TMP_C})
set(BASE_SRCS)
foreach(file ${TMP_SRCS})
    set(BASE_SRCS ${BASE_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FASTLIB_SRCS ${FASTLIB_SRCS} ${BASE_SRCS} PARENT_SCOPE )
add_library( base ${BASE_SRCS} )

add_dependencies( base template_types_detect )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/base)
