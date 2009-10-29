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
set( OPTPP_LIBS opt newmat ) #! FLIBS?

add_definitions( -DSETUP_C_SUBSCRIPTS -DUSING_DOUBLE -DHAVE_STD -DHAVE_NAMESPACES )
add_library( fastlib_optpp
    optimizer.h
)
set_target_properties( fastlib_optpp PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries( fastlib_optpp ${OPTPP_LIBS} )

add_executable( foo foo.cc)
target_link_libraries( foo fastlib_optpp ) #! ${OPTPP_LIBS} )

add_executable( optimizer_tests optimizer_tests.cc)
target_link_libraries( optimizer_tests fastlib_optpp ) #! ${OPTPP_LIBS} )
