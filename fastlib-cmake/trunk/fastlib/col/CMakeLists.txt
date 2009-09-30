cmake_minimum_required(VERSION 2.6)

include_directories( $ENV{FASTLIBPATH}		     
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

set( TMP_HEADERS 
    arraylist.h  col_string.h  fastalloc.h	heap.h	intmap.h  queue.h  rangeset.h
)
set( TMP_CXX col.cc )	
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(COL_SRCS)
foreach(file ${TMP_SRCS})
  set(COL_SRCS ${COL_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FASTLIB_SRCS ${FASTLIB_SRCS} ${COL_SRCS} PARENT_SCOPE )
add_library( col ${COL_SRCS} )
add_dependencies( col base )

add_executable( col_test
		col_test.cc
)

target_link_libraries( col_test
		       col
		       base	#needed?
)
install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/col)
