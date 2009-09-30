cmake_minimum_required(VERSION 2.6)

## include_directories( $ENV{FASTLIBPATH}		     
##                      ${CMAKE_CURRENT_BINARY_DIR}/../base/config
## )

set( TMP_HEADERS 
    lbfgs.h lbfgs_impl.h optimization_utils.h
)
set( TMP_CXX 
    
)	
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(lbfgs_SRCS)
foreach(file ${TMP_SRCS})
  set(lbfgs_SRCS ${lbfgs_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

## set(MLPACK_SRCS ${MLPACK_SRCS} ${lbfgs_SRCS} PARENT_SCOPE )
add_library( lbfgs ${lbfgs_SRCS} )
set_target_properties( lbfgs PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( lbfgs
    ${MLPACK_STD_LIBS}
)

#add_executable( lbfgs_test
#		test_lbfgs.cc
#)
#target_link_libraries( lbfgs_test
#    lbfgs
#    # mvu  #TODO - test depends on a whole algorithm??
#    ${MLPACK_STD_LIBS}
#)
## install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/lbfgs)
