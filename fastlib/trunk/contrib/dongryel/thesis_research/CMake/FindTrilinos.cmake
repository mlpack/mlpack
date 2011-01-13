# This will find out if a user has Trilinos installed.
# To pick only specific libraries (recommended), set ${TRILINOS_REQUIRED_LIBS}
# before calling this.
# It sets TRILINOS_INCLUDE_DIR and TRILINOS_LIBS and also finds MPI_INCLUDE_DIR

## Location of trilinos
set(TRILINOS_LIB_DIR /usr/local/lib CACHE PATH
  "Directory where trilinos is installed.")

## library 
if(NOT "${TRILINOS_REQUIRED_LIBS}")  # some defaults
   set(TRILINOS_REQUIRED_LIBS epetra anasazi aztecoo ifpack teuchos)
endif()

set(TRILINOS_LIBS)
foreach(lib ${TRILINOS_REQUIRED_LIBS})
    set(trilinos_lib trilinos_lib-NOTFOUND CACHE INTERNAL "")
    find_library(trilinos_lib ${lib}
        PATHS 
        ${TRILINOS_LIB_DIR}
        /opt/trilinos/lib 
    )
    if(NOT trilinos_lib)
        # try looking for libtrilinos_${lib}
        find_library(trilinos_lib trilinos_${lib}
            PATHS
            ${TRILINOS_LIB_DIR}
            /opt/trilinos/lib
        )
        if(NOT trilinos_lib)
            message(FATAL_ERROR "Trilinos library ${lib} not found.")
        endif()
    endif()
    set(TRILINOS_LIBS ${TRILINOS_LIBS} ${trilinos_lib})
endforeach()

## include dirs
find_path(TRILINOS_INCLUDE_DIR Trilinos_version.h 
    PATHS
    ${TRILINOS_LIB_DIR}/../include 
    /opt/trilinos/include
    PATH_SUFFIXES
    trilinos
)

if(TRILINOS_INCLUDE_DIR AND TRILINOS_LIBS)
    set(TRILINOS_FOUND "YES")
    message(STATUS "Found trilinos libraries.")
    message(STATUS "Found trilinos headers in ${TRILINOS_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Couldn't find trilinos.")
endif()
mark_as_advanced(TRILINOS_INCLUDE_DIR trilinos_lib)

find_path(MPI_INCLUDE_DIR mpi.h
    PATHS
    $ENV{HOME}/local/openmpi-1.4.3/include
    PATH_SUFFIXES
    mpi
    openmpi
)
if(MPI_INCLUDE_DIR)
    set(MPI_FOUND "YES")
    message(STATUS "Found MPI headers in ${MPI_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Couldn't find MPI headers.")
endif()
