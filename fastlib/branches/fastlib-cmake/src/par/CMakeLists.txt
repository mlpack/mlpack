cmake_minimum_required(VERSION 2.6)

include_directories( $ENV{FASTLIBPATH}		     
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

set( TMP_HEADERS grain.h task.h thread.h)
set( TMP_CXX thread.cc )
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(PAR_SRCS)
foreach(file ${TMP_SRCS})
  set(PAR_SRCS ${PAR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FASTLIB_SRCS ${FASTLIB_SRCS} ${PAR_SRCS} PARENT_SCOPE )
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ./)
add_library( par ${PAR_SRCS} )

add_dependencies( par col base )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/par)
