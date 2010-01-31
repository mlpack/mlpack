cmake_minimum_required(VERSION 2.6)

INCLUDE_DIRECTORIES( $ENV{FASTLIBPATH}
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

SET( FILE_SOURCE_FILES 
     textfile.cc
     textfile.h
)

ADD_LIBRARY ( file ${FILE_SOURCE_FILES}
)

ADD_EXECUTABLE( textfile_test
		textfile_test.cc
)

TARGET_LINK_LIBRARIES( textfile_test
		       base
                       file
)
