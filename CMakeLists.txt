cmake_minimum_required(VERSION 2.8)
project(mlpack C CXX)

# First, define all the compilation options.
# We default to debugging mode for developers.
option(DEBUG "Compile with debugging information" ON)
option(PROFILE "Compile with profiling information" ON)
option(ARMA_EXTRA_DEBUG "Compile with extra Armadillo debugging symbols." OFF)

# This is as of yet unused.
#option(PGO "Use profile-guided optimization if not a debug build" ON)

# Set the CFLAGS and CXXFLAGS depending on the options the user specified.
# Only GCC-like compilers support -Wextra, and other compilers give tons of
# output for -Wall, so only -Wall and -Wextra on GCC.
if(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
endif(CMAKE_COMPILER_IS_GNUCC)

# Debugging CFLAGS.  Turn optimizations off; turn debugging symbols on.
if(DEBUG)
  add_definitions(-DDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -O0")
else()
  add_definitions(-DARMA_NO_DEBUG)
  add_definitions(-DNDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O2")
endif(DEBUG)

# Profiling CFLAGS.  Turn profiling information on.
if(PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif(PROFILE)

# If the user asked for extra Armadillo debugging output, turn that on.
if(ARMA_EXTRA_DEBUG)
  add_definitions(-DARMA_EXTRA_DEBUG)
endif(ARMA_EXTRA_DEBUG)

# Now, find the libraries we need to compile against.  Several variables can be
# set to manually specify the directory in which each of these libraries
# resides.
#   ARMADILLO_LIBRARY - location of libarmadillo.so / armadillo.lib
#   ARMADILLO_INCLUDE_DIR - directory containing <armadillo>
#   LAPACK_cheev_LIBRARY - location of LAPACK
#   LIBXML2_INCLUDE_DIR - location of LibXml2 includes
#   LIBXML2_LIBRARIES - locations of libxml2.so or libxml2.lib
#   BOOST_ROOT - root of Boost installation
#   BOOST_INCLUDEDIR - include directory for Boost
#   BOOST_LIBRARYDIR - library directory for Boost

# set path right
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# LAPACK finds BLAS as a dependency.
find_package(LAPACK REQUIRED)
find_package(Armadillo 2.4.0 REQUIRED)
find_package(LibXml2 REQUIRED)

# Include directories for the previous dependencies.
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${LIBXML2_INCLUDE_DIR})

# Pthreads may have a compiler definition.
add_definitions(${PTHREADS_DEFINITIONS})

# Unfortunately this configuration variable is necessary and will need to be
# updated as time goes on and new versions are released.
set(Boost_ADDITIONAL_VERSIONS
  "1.41" "1.41.0" "1.42" "1.42.0" "1.43" "1.43.0" "1.44" "1.44.0" "1.45.0"
  "1.46.0" "1.46.1" "1.47.0" "1.48.0" "1.49.0")
find_package(Boost
    COMPONENTS
      math_c99
      program_options
      unit_test_framework
    REQUIRED
)
include_directories(${Boost_INCLUDE_DIRS})

# Save the actual link paths (because they will get overwritten if we discover
# we need to find Boost.Random too).
set(Boost_BACKUP_LIBRARIES ${Boost_LIBRARIES})

# We need to include Boost.Random, but only if newer than 1.45 (as of 1.46 it
# became a separate package with its own linkable library object).
if(Boost_MAJOR_VERSION EQUAL 1 AND Boost_MINOR_VERSION GREATER 45)
  find_package(Boost
      COMPONENTS
          random
      REQUIRED
  )

  # Restore actual link locations of the other Boost libraries.
  set(Boost_LIBRARIES ${Boost_LIBRARIES} ${Boost_BACKUP_LIBRARIES})

  # This may be redundant.
  include_directories(${Boost_INCLUDE_DIRS})

endif(Boost_MAJOR_VERSION EQUAL 1 AND Boost_MINOR_VERSION GREATER 45)

# For Boost testing framework (will have no effect on non-testing executables).
# This specifies to Boost that we are dynamically linking to the Boost test
# library.
add_definitions(-DBOOST_TEST_DYN_LINK)

# Create a 'distclean' target in case the user is using an in-source build for
# some reason.
include(CMake/TargetDistclean.cmake OPTIONAL)

include_directories(${CMAKE_SOURCE_DIR})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

# Recurse into the rest of the project.
add_subdirectory(src/mlpack)

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
          kernel_pca kmeans lars linear_regression local_coordinate_coding mvu
          nbc nca pca radical sparse_coding
        COMMENT "Generating man pages from built executables."
    )

    # Set the rules to install the documentation.
    install(DIRECTORY ${CMAKE_BINARY_DIR}/share/man/
        DESTINATION share/man/man1/)
  endif (NOT TXT2MAN)
endif (UNIX)
