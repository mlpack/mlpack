cmake_minimum_required(VERSION 2.6)

include_directories( $ENV{FASTLIBPATH}		     
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

set( TMP_HEADERS 
    sparse_matrix.h
    sparse_matrix_impl.h
)
set( TMP_CXX )
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(sparse_SRCS)
foreach(file ${TMP_SRCS})
  set(sparse_SRCS ${sparse_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FASTLIB_SRCS ${FASTLIB_SRCS} ${sparse_SRCS} PARENT_SCOPE )
add_library( sparse ${sparse_SRCS} ) 
set_target_properties( sparse PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies( sparse 
    trilinos base
)
target_link_libraries( sparse 
    trilinos base
)

add_executable( sparse_test
    sparse_matrix_test.cc
)
set_target_properties( kernel_pca_test PROPERTIES 
    COMPILE_FLAGS -fexceptions #req. for trilinos; portable?
)
target_link_libraries( sparse_test sparse )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/sparse)
