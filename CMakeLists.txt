cmake_minimum_required(VERSION 2.8)
project(mlpack C CXX)

## External Libraries
#  ls /usr/share/cmake-2.6/Modules/Find* | \
#  perl -ne 's#.*Modules/Find(.*)>cmake#\1#; print'

# set path right
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

find_package(LAPACK REQUIRED)  # LAPACK finds BLAS as a dependency
find_package(Pthreads REQUIRED)
find_package(Armadillo 2.4.0 REQUIRED)
find_package(LibXml2 REQUIRED)

# Include directories for the previous dependencies.
include_directories(${PTHREADS_INCLUDE_DIR})
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Pthreads may have a compiler definition.
add_definitions(${PTHREADS_DEFINITIONS})

# libxml2 requires some special handling
find_package(PkgConfig)

pkg_check_modules(PC_LIBXML QUIET libxml-2.0)
set(LIBXML2_DEFINITIONS ${PC_LIBXML_CFLAGS_OTHER})

find_path(LIBXML2_INCLUDE_DIR libxml/xpath.h
          HINTS ${PC_LIBXML_INCLUDEDIR} ${PC_LIBXML_INCLUDE_DIRS}
                    PATH_SUFFIXES libxml2 )

find_library(LIBXML2_LIBRARY NAMES xml2 libxml2
             HINTS ${PC_LIBXML_LIBDIR} ${PC_LIBXML_LIBRARY_DIRS} )

set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARY})
set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(LibXml2  DEFAULT_MSG
                                  LIBXML2_LIBRARY LIBXML2_INCLUDE_DIR)

mark_as_advanced(LIBXML2_INCLUDE_DIR LIBXML2_LIBRARY)

include_directories(${LIBXML2_INCLUDE_DIR})

# Unfortunately this configuration variable is necessary and will need to be
# updated as time goes on and new versions are released.
set(Boost_ADDITIONAL_VERSIONS
  "1.41" "1.41.0" "1.42" "1.42.0" "1.43" "1.43.0"
  "1.44" "1.44.0" "1.45.0")
set(BOOST_LIBS "boost_math_c99" "boost_program_options")
find_package(Boost COMPONENTS
  math_c99
  program_options
  unit_test_framework
  REQUIRED) # May require math_tr1?
include_directories(${Boost_INCLUDE_DIRS})

# We need to include Boost random, but only if newer than 1.45 (as of 1.46 it
# became a separate package with its own linkable library object).
if(Boost_MAJOR_VERSION EQUAL 1 AND Boost_MINOR_VERSION GREATER 45)
  find_package(Boost COMPONENTS
  random
  REQUIRED)
  set(BOOST_LIBS ${BOOST_LIBS} "boost_random")
  # This may be redundant.
  include_directories(${Boost_INCLUDE_DIRS})
endif(Boost_MAJOR_VERSION EQUAL 1 AND Boost_MINOR_VERSION GREATER 45)

# Default to debugging mode for developers.
option(DEBUG "Compile with debugging information" ON)
option(PROFILE "Compile with profiling information" ON)
option(ARMA_EXTRA_DEBUG "Compile with extra Armadillo debugging symbols." OFF)

# This is as of yet unused.
#option(PGO "Use profile-guided optimization if not a debug build" ON)

if(DEBUG)
  add_definitions(-DDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O0 -Wextra")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -Wall -Wextra -O0")
else()
  add_definitions(-DARMA_NO_DEBUG)
  add_definitions(-DNDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -Wall -Wextra -O2")
endif(DEBUG)

if(PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(PROFILE)

if(ARMA_EXTRA_DEBUG)
  add_definitions(-DARMA_EXTRA_DEBUG)
endif(ARMA_EXTRA_DEBUG)

# For Boost testing framework (will have no effect on non-testing executables).
# This specifies to Boost that we are dynamically linking to the Boost test
# library.

add_definitions(-DBOOST_TEST_DYN_LINK)

# libxml2 requires special handling
include_directories(${LIBXML2_INCLUDE_DIRS})

# distclean option because cmake doesn't support it
include(CMake/TargetDistclean.cmake OPTIONAL)

include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

## recurse
add_subdirectory(src/mlpack) # mlpack

# Make a target to generate the documentation.  If Doxygen isn't installed, then
# I guess this option will just be unavailable.
find_package(Doxygen)
if (DOXYGEN_FOUND)
  # It may be better if we built the documentation into the build directory, but
  # I'm leaving this for a later day.
  add_custom_target(doc
    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
  )
  add_custom_command(TARGET doc
    POST_BUILD
    COMMAND cmake -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/doc ${CMAKE_BINARY_DIR}/doc
    COMMENT "Copying API documentation to build directory"
  )

  install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/html
    DESTINATION share/doc/mlpack
    COMPONENT doc
    OPTIONAL
  )
endif (DOXYGEN_FOUND)

# Make a target to generate the man page documentation, but only if we are on a
# UNIX-like system.
if (UNIX)
  find_program(TXT2MAN txt2man)

  # It's not a requirement that we make man pages.
  if (NOT TXT2MAN)
    message(WARNING "txt2man not found; man pages will not be generated.")
  else (NOT TXT2MAN)
    # We have the tools.  We can make them.
    add_custom_target(man ALL
      ${CMAKE_CURRENT_SOURCE_DIR}/CMake/allexec2man.sh
          ${CMAKE_CURRENT_SOURCE_DIR}/CMake/exec2man.sh
          ${CMAKE_BINARY_DIR}/share/man
      WORKING_DIRECTORY
          ${CMAKE_BINARY_DIR}/bin
      DEPENDS
          allkfn allknn emst gmm hmm_generate hmm_loglik hmm_train hmm_viterbi
          kernel_pca kmeans lars linear_regression local_coordinate_coding nbc
          nca pca radical sparse_coding
      COMMENT "Generating man pages from built executables."
    )

    # Set the rules to install the documentation.
    install(DIRECTORY ${CMAKE_BINARY_DIR}/share/man/
        DESTINATION share/man/man1/)
  endif (NOT TXT2MAN)
endif (UNIX)
