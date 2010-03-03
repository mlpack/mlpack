cmake_minimum_required(VERSION 2.8)

## Location of trilinos
set(TRILINOS_BASE_DIR /usr/local CACHE PATH
  "Directory where trilinos is installed." )
#set(TRILINOS_LIB_DIR ${TRILINOS_BASE_DIR}/lib )

## library 
#set(TRILINOS_REQUIRED_LIBS epetra anasazi aztecoo ifpack kokkos teuchos)
set(TRILINOS_REQUIRED_LIBS epetra anasazi aztecoo ifpack teuchos)
set(TRILINOS_LIBS)
foreach(lib ${TRILINOS_REQUIRED_LIBS})
    set(trilinos_lib trilinos_lib-NOTFOUND CACHE INTERNAL "")
    find_library(trilinos_lib ${lib}
        PATHS 
        ${TRILINOS_BASE_DIR}/lib 
        /opt/trilinos/lib 
    )
    if(NOT trilinos_lib)
        # try looking for libtrilinos_${lib}
        find_library(trilinos_lib trilinos_${lib}
            PATHS
            ${TRILINOS_BASE_DIR}/lib
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
    ${TRILINOS_BASE_DIR}/include 
    /opt/trilinos/include
    PATH_SUFFIXES
    trilinos
)

if(TRILINOS_INCLUDE_DIR AND TRILINOS_LIBS)
    set(TRILINOS_FOUND "YES")
else()
    message(FATAL_ERROR "Couldn't find trilinos.")
endif()
mark_as_advanced(TRILINOS_INCLUDE_DIR trilinos_lib)

find_path(MPI_INCLUDE_DIR mpi.h
    PATHS
    /opt/openmpi/include
    PATH_SUFFIXES
    mpi
    openmpi
)
if(MPI_INCLUDE_DIR)
    set(MPI_FOUND "YES")
else()
    message(FATAL_ERROR "Couldn't find MPI headers.")
endif()

include_directories( ${TRILINOS_INCLUDE_DIR} ${MPI_INCLUDE_DIR} )
## libs
#link_directories( ${TRILINOS_LIB_DIR} )
#set(TRILINOS_LIBS ${TRILINOS_LIBS})
#set(TRILINOS_LIBS tpetra anasazi aztecoo ifpack kokkos teuchos)



# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   sparse_matrix.h
   sparse_matrix_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at the parent scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_library(sparse ${SOURCES})
# link dependencies of sparse
target_link_libraries(sparse
   base
   ${TRILINOS_LIBS}
   lapack
)

# test executable
add_executable(sparse_test
   sparse_matrix_test.cc
)
set_target_properties(sparse_test PROPERTIES
   COMPILE_FLAGS -fexceptions # required for trilinos
)
# link dependencies of test executable
target_link_libraries(sparse_test
   sparse
   ${TRILINOS_LIBS}
)
