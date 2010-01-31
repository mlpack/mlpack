cmake_minimum_required(VERSION 2.6)

## include_directories( $ENV{FASTLIBPATH}		     
##                      ${CMAKE_CURRENT_BINARY_DIR}/../base/config
## )

set( TMP_HEADERS 
    kernel_pca.h
    kernel_pca_impl.h
)
set( TMP_CXX 
    
)	
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(kernel_pca_SRCS)
foreach(file ${TMP_SRCS})
  set(kernel_pca_SRCS ${kernel_pca_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

## set(MLPACK_SRCS ${MLPACK_SRCS} ${kernel_pca_SRCS} PARENT_SCOPE )
add_library( kernel_pca ${kernel_pca_SRCS} )
set_target_properties( kernel_pca PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( kernel_pca
    # fastlib/sparse #TODO
    ${MLPACK_STD_LIBS}
)

add_executable( kernel_pca_test
		kernel_pca_test.cc
)
set_target_properties( kernel_pca_test PROPERTIES 
    COMPILE_FLAGS -fexceptions #portable?
)
target_link_libraries( kernel_pca_test
    kernel_pca
    ${MLPACK_STD_LIBS}
)
## install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/kernel_pca)
