cmake_minimum_required(VERSION 2.6)

## include_directories( $ENV{FASTLIBPATH}		     
##                      ${CMAKE_CURRENT_BINARY_DIR}/../base/config
## )

set( TMP_HEADERS 
    ridge_regression.h
    ridge_regression_impl.h
    ridge_regression_util.h    
)
set( TMP_CXX 

)	
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(ridge_regression_SRCS)
foreach(file ${TMP_SRCS})
  set(ridge_regression_SRCS ${ridge_regression_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

## set(MLPACK_SRCS ${MLPACK_SRCS} ${ridge_regression_SRCS} PARENT_SCOPE )
add_library( ridge_regression ${ridge_regression_SRCS} )
set_target_properties( ridge_regression PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( ridge_regression
    quicsvd
    ${MLPACK_STD_LIBS}
)

add_executable( ridge_regression_test
		ridge_regression_test.cc
)
target_link_libraries( ridge_regression_test
    ridge_regression
#   ${MLPACK_STD_LIBS}
)

add_executable( ridge
		ridge_main.cc
)
target_link_libraries( ridge
    ridge_regression
    ${MLPACK_STD_LIBS}
)
## install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/ridge_regression)
