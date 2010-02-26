cmake_minimum_required(VERSION 2.6)

include_directories( $ENV{FASTLIBPATH}
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

set( TMP_HEADERS 
    datanode.h  fx.h  stopwatch.h
)
set( TMP_CXX )
set( TMP_C 
    datanode.c  fx.c  stopwatch.c
)
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX} ${TMP_C})
set(FX_SRCS)
foreach(file ${TMP_SRCS})
    set(FX_SRCS ${FX_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FASTLIB_SRCS ${FASTLIB_SRCS} ${FX_SRCS} PARENT_SCOPE )
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ./)
add_library( fx ${FX_SRCS} )

add_dependencies( fx base )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/fx)
