cmake_minimum_required(VERSION 2.6)

## include_directories( $ENV{FASTLIBPATH}		     
##                      ${CMAKE_CURRENT_BINARY_DIR}/../base/config
## )

set( TMP_HEADERS 
   allkfn.h 
)
set( TMP_CXX 
    
)	
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(allkfn_SRCS)
foreach(file ${TMP_SRCS})
  set(allkfn_SRCS ${allkfn_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

## set(MLPACK_SRCS ${MLPACK_SRCS} ${allkfn_SRCS} PARENT_SCOPE )
add_library( allkfn ${allkfn_SRCS} )
set_target_properties( allkfn PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( allkfn
    ${MLPACK_STD_LIBS}
)

add_executable( allkfn_test
		allkfn_test.cc
)
target_link_libraries( allkfn_test
    allkfn
    ${MLPACK_STD_LIBS}
)

add_executable( allkfn_exe
		main.cc
)
target_link_libraries( allkfn_exe
    allkfn
    ${MLPACK_STD_LIBS}
)

## install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/allkfn)
