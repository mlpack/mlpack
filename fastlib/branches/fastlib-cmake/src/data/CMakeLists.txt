cmake_minimum_required(VERSION 2.6)

include_directories( $ENV{FASTLIBPATH}		     
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

set( TMP_HEADERS 
    crossvalidation.h  dataset.h
)
set( TMP_CXX 
    dataset.cc
)
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})

set(DATA_SRCS)
foreach(file ${TMP_SRCS})
  set(DATA_SRCS ${DATA_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DATA_SRCS} PARENT_SCOPE )
add_library( data ${DATA_SRCS} )

target_link_libraries( data
		       base file la
)

add_executable( dataset_test
		dataset_test.cc
)

target_link_libraries( dataset_test
		       data fx math
)

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/data)
