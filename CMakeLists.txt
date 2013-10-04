cmake_minimum_required(VERSION 2.8.5)
project(mlpack C CXX)

# First, define all the compilation options.
# We default to debugging mode for developers.
option(DEBUG "Compile with debugging information" OFF)
option(PROFILE "Compile with profiling information" OFF)
option(ARMA_EXTRA_DEBUG "Compile with extra Armadillo debugging symbols." OFF)
option(MATLAB_BINDINGS "Compile MATLAB bindings if MATLAB is found." OFF)

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
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3")
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
#   LIBXML2_INCLUDE_DIR - location of LibXml2 includes
#   LIBXML2_LIBRARIES - locations of libxml2.so or libxml2.lib
#   BOOST_ROOT - root of Boost installation
#   BOOST_INCLUDEDIR - include directory for Boost
#   BOOST_LIBRARYDIR - library directory for Boost

# set path right
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

find_package(Armadillo 3.6.0 REQUIRED)

# If Armadillo was compiled without ARMA_64BIT_WORD and we are on a 64-bit
# system (where size_t will be 64 bits), suggest to the user that they should
# compile Armadillo with 64-bit words.
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  # Can we open the configuration file?  If not, issue a warning.
  if(NOT EXISTS "${ARMADILLO_INCLUDE_DIRS}/armadillo_bits/config.hpp")
    message(WARNING "Armadillo configuration file "
        "(${ARMADILLO_INCLUDE_DIRS}/armadillo_bits/config.hpp) does not exist!")
  else(NOT EXISTS "${ARMADILLO_INCLUDE_DIRS}/armadillo_bits/config.hpp")
    # We are on a 64-bit system.  Does Armadillo have ARMA_64BIT_WORD enabled?
    file(READ "${ARMADILLO_INCLUDE_DIRS}/armadillo_bits/config.hpp" ARMA_CONFIG)
    string(REGEX MATCH
        "[\r\n][ ]*#define ARMA_64BIT_WORD"
        ARMA_HAS_64BIT_WORD_PRE
        "${ARMA_CONFIG}")

    string(LENGTH "${ARMA_HAS_64BIT_WORD_PRE}" ARMA_HAS_64BIT_WORD)

    if(ARMA_HAS_64BIT_WORD EQUAL 0)
      message(WARNING "This is a 64-bit system, but Armadillo was compiled "
          "without 64-bit index support.  Consider recompiling Armadillo with "
          "ARMA_64BIT_WORD to enable 64-bit indices (large matrix support). "
          "MLPACK will still work without ARMA_64BIT_WORD defined, but will not "
          "scale to matrices with more than 4 billion elements.")
    endif(ARMA_HAS_64BIT_WORD EQUAL 0)
  endif(NOT EXISTS "${ARMADILLO_INCLUDE_DIRS}/armadillo_bits/config.hpp")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)

# On Windows, Armadillo should be using LAPACK and BLAS but we still need to
# link against it.  We don't want to use the FindLAPACK or FindBLAS modules
# because then we are required to have a FORTRAN compiler (argh!) so we will try
# and find LAPACK and BLAS ourselves, using a slightly modified variant of the
# script Armadillo uses to find these.
if (WIN32)
  find_library(LAPACK_LIBRARY
      NAMES lapack liblapack lapack_win32_MT lapack_win32
      PATHS "C:/Program Files/Armadillo"
      PATH_SUFFIXES "examples/lib_win32/")

  if (NOT LAPACK_LIBRARY)
    message(FATAL_ERROR "Cannot find LAPACK library (.lib)!")
  endif (NOT LAPACK_LIBRARY)

  find_library(BLAS_LIBRARY
      NAMES blas libblas blas_win32_MT blas_win32
      PATHS "C:/Program Files/Armadillo"
      PATH_SUFFIXES "examples/lib_win32/")

  if (NOT BLAS_LIBRARY)
    message(FATAL_ERROR "Cannot find BLAS library (.lib)!")
  endif (NOT BLAS_LIBRARY)

  # Piggyback LAPACK and BLAS linking into Armadillo link.
  set(ARMADILLO_LIBRARIES
      "${ARMADILLO_LIBRARIES};${BLAS_LIBRARY};${LAPACK_LIBRARY}")
endif (WIN32)

find_package(LibXml2 2.6.0 REQUIRED)

# On Windows, LibXml2 has a couple dependencies and we want to make sure they
# exist.  We don't need the include directory, so we just use a find_library
# call, looking for .dlls.
if (WIN32)
  # Find a .dll, not a .lib, because libxml2 is probably dynamically linked.
  set(CMAKE_OLD_SUFFIXES "${CMAKE_FIND_LIBRARY_SUFFIXES}")
  set(CMAKE_FIND_LIBRARY_SUFFIXES ".dll")

  find_library(ZLIB_LIBRARY
      zlib1
      HINTS "[HKEY_LOCAL_MACHINE\\SOFTWARE\\GnuWin32\\ZLib:InstallPath]/lib"
  )
  if (NOT ZLIB_LIBRARY)
    message(WARNING "zlib1.dll not found!  libxml2.dll may depend on this, and
      if it is not present MLPACK may not work.")
  endif (NOT ZLIB_LIBRARY)

  find_library(ICONV_LIBRARY iconv)
  if (NOT ICONV_LIBRARY)
    message(WARNING "iconv.dll not found!  libxml2.dll may depend on this, and
      if it is not present MLPACK may not work.")
  endif (NOT ICONV_LIBRARY)

  find_file(ICONV_HEADER
      iconv.h
      HINTS ${LIBXML2_INCLUDE_DIR})
  if (NOT ICONV_HEADER)
    message(FATAL_ERROR "iconv.h not found!  It can be placed in a standard
        include directory or the LibXml2 include directory.")
  endif (NOT ICONV_HEADER)
  # Add the directory containing iconv.h to the include directories.
  get_filename_component(ICONV_H_DIR ${ICONV_HEADER} PATH)
  include_directories(${ICONV_H_DIR})

  # Reset suffixes.
  set(CMAKE_FIND_LIBRARY_SUFFIXES "${CMAKE_OLD_SUFFIXES}")
endif (WIN32)

# Include directories for the previous dependencies.
include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(${LIBXML2_INCLUDE_DIR})

# Unfortunately this configuration variable is necessary and will need to be
# updated as time goes on and new versions are released.
set(Boost_ADDITIONAL_VERSIONS
  "1.41" "1.41.0" "1.42" "1.42.0" "1.43" "1.43.0" "1.44" "1.44.0" "1.45.0"
  "1.46.0" "1.46.1" "1.47.0" "1.48.0" "1.49.0")
find_package(Boost
    COMPONENTS
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
link_directories(${Boost_LIBRARY_DIRS})

# On Windows, automatic linking is performed, so we don't need to worry about
# it.  Clear the list of libraries to link against and let Visual Studio handle
# it.
if (WIN32)
  link_directories(${Boost_LIBRARY_DIRS})
  set(Boost_LIBRARIES "")
endif (WIN32)

# For Boost testing framework (will have no effect on non-testing executables).
# This specifies to Boost that we are dynamically linking to the Boost test
# library.
add_definitions(-DBOOST_TEST_DYN_LINK)

# Create a 'distclean' target in case the user is using an in-source build for
# some reason.
include(CMake/TargetDistclean.cmake OPTIONAL)

include_directories(${CMAKE_SOURCE_DIR})

# On Windows, things end up under Debug/ or Release/.
if (WIN32)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else (WIN32)
  # If not on Windows, put them under more standard UNIX-like places.  This is
  # necessary, otherwise they would all end up in
  # ${CMAKE_BINARY_DIR}/src/mlpack/methods/... or somewhere else random like
  # that.
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
endif (WIN32)

# Recurse into the rest of the project.
add_subdirectory(src/mlpack)

# Make a target to generate the documentation.  If Doxygen isn't installed, then
# I guess this option will just be unavailable.
find_package(Doxygen)
if (DOXYGEN_FOUND)
  # Preprocess the Doxyfile.  This is done before 'make doc'.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/Doxyfile
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND} -D DESTDIR=${CMAKE_BINARY_DIR} -P
          ${CMAKE_CURRENT_SOURCE_DIR}/CMake/GenerateDoxyfile.cmake
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "Creating Doxyfile to generate Doxygen documentation"
  )

  # Generate documentation.
  add_custom_target(doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_BINARY_DIR}/Doxyfile
      DEPENDS ${CMAKE_BINARY_DIR}/Doxyfile
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating API documentation with Doxygen"
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
          allkfn allknn allkrann cf det emst fastmks gmm hmm_generate hmm_loglik
          hmm_train hmm_viterbi kernel_pca kmeans lars linear_regression
          local_coordinate_coding nbc nca nmf pca radical range_search
          sparse_coding
        COMMENT "Generating man pages from built executables."
    )

    # Set the rules to install the documentation.
    install(DIRECTORY ${CMAKE_BINARY_DIR}/share/man/
        DESTINATION share/man/man1/)
  endif (NOT TXT2MAN)
endif (UNIX)
