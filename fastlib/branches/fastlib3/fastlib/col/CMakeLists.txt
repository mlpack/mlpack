cmake_minimum_required(VERSION 2.6)

INCLUDE_DIRECTORIES( $ENV{FASTLIBPATH}		     
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

ADD_LIBRARY( col
	     col.cc
	     arraylist.h
	     col_string.h
	     fastalloc.h
	     heap.h
	     intmap.h
	     queue.h
	     rangeset.h
)

ADD_EXECUTABLE( col_test
		col_test.cc
)

TARGET_LINK_LIBRARIES( col_test
		       col
		       base
)
