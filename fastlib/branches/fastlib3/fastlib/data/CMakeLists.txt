cmake_minimum_required(VERSION 2.6)

INCLUDE_DIRECTORIES( $ENV{FASTLIBPATH}		     
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

ADD_LIBRARY( data
	     dataset.cc
	     dataset.h
	     crossvalidation.h
)

ADD_EXECUTABLE( dataset_test
		dataset_test.cc
)

TARGET_LINK_LIBRARIES( dataset_test
		       base
		       col
		       data
		       file
)
