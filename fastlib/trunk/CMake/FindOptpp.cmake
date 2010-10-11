# This will find out if a user has Optpp installed.
# This is a hack originally written by Leif Poorman.

## Location of opt++
set(OPTPP_BASE_DIR /usr/local CACHE PATH
  "Directory where opt++ is installed." )
set(OPTPP_LIB_DIR ${OPTPP_BASE_DIR}/lib )
#set(OPTPP_INCLUDE_DIR ${OPTPP_BASE_DIR}/include )
## library 
find_library(OPTPP_LIB opt 
    PATHS ${OPTPP_LIB_DIR} /opt/optpp-2.4/lib ) #!
find_library(NEWMAT_LIB newmat 
    PATHS ${OPTPP_LIB_DIR} /opt/optpp-2.4/lib ) #!
## include dirs
find_path(OPTPP_INCLUDE_DIR Opt.h  #! kind of common...
    PATHS
    ${OPTPP_BASE_DIR}/include 
    /opt/optpp-2.4/include
)
message(STATUS "optpp_lib ${OPTPP_LIB} newmat_lib ${NEWMAT_LIB} include_dir ${OPTPP_INCLUDE_DIR}")
if(${NEWMAT_LIB} AND ${OPTPP_LIB})
    if(${OPTPP_INCLUDE_DIR})
        set(OPTPP_FOUND "YES")
    endif()
endif()

set(OPTPP_LIBS ${OPTPP_LIB} ${NEWMAT_LIB} blas) #! FLIBS?

#if(NOT OPTPP_FOUND)
#  message(FATAL_ERROR "Could not find Opt++.")
#else()
#  message(STATUS "Found Opt++ headers in ${OPTPP_INCLUDE_DIR}")
#endif()
