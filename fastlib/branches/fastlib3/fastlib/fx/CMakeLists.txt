cmake_minimum_required(VERSION 2.6)

INCLUDE_DIRECTORIES( $ENV{FASTLIBPATH}
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

FILE( GLOB FX_SOURCE_FILES *.cc *.c *.h
)

ADD_LIBRARY ( fx ${FX_SOURCE_FILES}
)

ADD_DEPENDENCIES ( fx base
)
