cmake_minimum_required(VERSION 2.6)

ADD_SUBDIRECTORY( config )

INCLUDE_DIRECTORIES( config
                     $ENV{FASTLIBPATH}
                     $ENV{FASTLIBPATH}/fastlib
                     ${CMAKE_CURRENT_BINARY_DIR}/config
)

FILE( GLOB BASE_SOURCE_FILES *.cc *.h *.c )

ADD_LIBRARY ( base ${BASE_SOURCE_FILES} )

ADD_DEPENDENCIES ( base template_types_detect )
