cmake_minimum_required(VERSION 2.6)

## include_directories( $ENV{FASTLIBPATH}		     
##                      ${CMAKE_CURRENT_BINARY_DIR}/../base/config
## )

set( TMP_HEADERS 
    
)
set( TMP_CXX 
    
)	
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})
set(<{DIR}>_SRCS)
foreach(file ${TMP_SRCS})
  set(<{DIR}>_SRCS ${<{DIR}>_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

## set(MLPACK_SRCS ${MLPACK_SRCS} ${<{DIR}>_SRCS} PARENT_SCOPE )
add_library( <{DIR}> ${<{DIR}>_SRCS} )
## set_target_properties( <{DIR}> PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( <{DIR}>
    ...
    ${MLPACK_STD_LIBS}
)

add_executable( <{DIR}>_test
		<{DIR}>_test.cc
)
target_link_libraries( <{DIR}>_test
    <{DIR}>
    ${MLPACK_STD_LIBS}
)
## install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/<{DIR}>)
