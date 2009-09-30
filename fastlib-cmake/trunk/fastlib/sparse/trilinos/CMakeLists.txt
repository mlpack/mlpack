cmake_minimum_required(VERSION 2.6)

include_directories( $ENV{FASTLIBPATH}		     
                     ${CMAKE_CURRENT_BINARY_DIR}/../base/config
)

set( TMP_HEADERS 
    bounds.h
    kdtree.h
    kdtree_impl.h
    spacetree.h
    statistic.h
)
set( TMP_CXX )
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(TREE_SRCS)
foreach(file ${TMP_SRCS})
  set(TREE_SRCS ${TREE_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(FASTLIB_SRCS ${FASTLIB_SRCS} ${TREE_SRCS} PARENT_SCOPE )
add_library( tree ${TREE_SRCS} ) 
set_target_properties( tree PROPERTIES LINKER_LANGUAGE CXX)

add_dependencies( tree 
    base col data 
    file  #file_int 
    fx la
)
target_link_libraries( tree 
    base col data 
    file  #file_int 
    fx la
)

add_executable( tree_test
		tree_test.cc
)

target_link_libraries( tree_test tree )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/tree)
