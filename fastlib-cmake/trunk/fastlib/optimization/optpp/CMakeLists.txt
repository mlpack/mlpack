## Location of opt++
set(OPTPP_BASE_DIR /usr/local CACHE PATH
  "Directory where opt++ is installed." )
set(OPTPP_LIB_DIR ${OPTPP_BASE_DIR}/lib )
set(OPTPP_INCLUDE_DIR ${OPTPP_BASE_DIR}/include )
## library 
find_library(OPTPP_LIB opt ${OPTPP_LIB_DIR}) #!
find_library(OPTPP_LIB newmat ${OPTPP_LIB_DIR}) #!
## include dirs
include_directories( ${OPTPP_INCLUDE_DIR} )
set( OPTPP_LIBS opt newmat blas ) #! FLIBS?

#add_definitions( -DUSING_DOUBLE -DHAVE_STD -DHAVE_NAMESPACES -DSETUP_C_SUBSCRIPTS ) 
#add_definitions( -DDEBUG )

set( TMP_HEADERS 
    optimizer.h
)
set( TMP_CXX )
set( TMP_SRCS ${TMP_HEADERS} ${TMP_CXX})

set(OPTPP_SRCS)
foreach(file ${TMP_SRCS})
  set(OPTPP_SRCS ${OPTPP_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${OPTPP_SRCS} PARENT_SCOPE )

message("in opt++:")
message("TMP_SRCS: ${TMP_SRCS}")
message("OPTPP_SRCS: ${OPTPP_SRCS}")
message("FASTLIB_SRCS: ${FASTLIB_SRCS}")

add_library( fastlib_optpp ${OPTPP_SRCS} )
set_target_properties( fastlib_optpp PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( fastlib_optpp ${OPTPP_LIBS} )

install(FILES ${TMP_HEADERS} DESTINATION include/fastlib/optimization/optpp)

add_executable( optimizer_tests optimizer_tests.cc)
target_link_libraries( optimizer_tests fx base fastlib_optpp ) #! ${OPTPP_LIBS} )
