cmake_minimum_required(VERSION 2.6)

include_directories( $ENV{FASTLIBPATH}
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
) 				#TODO put at top level?

set( TMP_HEADERS textfile.h)
set( TMP_CXX textfile.cc)
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(FILE_SRCS)
foreach(file ${TMP_SRCS})
  set(FILE_SRCS ${FILE_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${FILE_SRCS} PARENT_SCOPE )

add_library( file ${FILE_SRCS} )
target_link_libraries( file base col )
#add_dependencies( file base col )

add_executable( textfile_test
		textfile_test.cc
)
target_link_libraries( textfile_test
		       file fx
)

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/file)
