cmake_minimum_required(VERSION 2.6)

## include_directories( $ENV{FASTLIBPATH}		     
##                      ${CMAKE_CURRENT_BINARY_DIR}/../base/config
## )

set( TMP_HEADERS 
    cosine_tree.h
    quicsvd.h   
)
set( TMP_CXX 
    cosine_tree.cc
    quicsvd.cc
)	
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(quicsvd_SRCS)
foreach(file ${TMP_SRCS})
  set(quicsvd_SRCS ${quicsvd_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

## set(MLPACK_SRCS ${MLPACK_SRCS} ${quicsvd_SRCS} PARENT_SCOPE )
add_library( quicsvd ${quicsvd_SRCS} )
## set_target_properties( quicsvd PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( quicsvd
    ${MLPACK_STD_LIBS}
)

add_executable( quicsvd_test
		quicsvd_test.cc
)
target_link_libraries( quicsvd_test
    quicsvd
    ${MLPACK_STD_LIBS}
)

add_executable( quicsvd_main
		quicsvd_main.cc
)
target_link_libraries( quicsvd_main
    quicsvd
    ${MLPACK_STD_LIBS}
)
## install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/quicsvd)
