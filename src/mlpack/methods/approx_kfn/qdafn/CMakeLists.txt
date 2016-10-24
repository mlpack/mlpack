# Much of this is borrowed from mlpack's CMakeLists.txt.
cmake_minimum_required(VERSION 2.8.5)
project(qdafn C CXX)

# Ensure that we have a C++11 compiler.
include(CXX11.cmake)
check_for_cxx11_compiler(HAS_CXX11)
if(NOT HAS_CXX11)
  message(FATAL_ERROR "No C++11 compiler available!")
endif()
enable_cxx11()

# Define compilation options.
option(DEBUG "Compile with debugging information" ON)
option(PROFILE "Compile with profiling information" ON)

# Set the CFLAGS and CXXFLAGS depending on the options the user specified.
# Only GCC-like compilers support -Wextra, and other compilers give tons of
# output for -Wall, so only -Wall and -Wextra on GCC.
if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ftemplate-depth=1000")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
endif()

# If using clang, we have to link against libc++ depending on the
# OS (at least on some systems). Further, gcc sometimes optimizes calls to
# math.h functions, making -lm unnecessary with gcc, but it may still be
# necessary with clang.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (APPLE)
    # detect OS X version. Use '/usr/bin/sw_vers -productVersion' to
    # extract V from '10.V.x'.)
    exec_program(/usr/bin/sw_vers ARGS
        -productVersion OUTPUT_VARIABLE MACOSX_VERSION_RAW)
    string(REGEX REPLACE
        "10\\.([0-9]+).*" "\\1"
        MACOSX_VERSION
        "${MACOSX_VERSION_RAW}")

     # OSX Lion (10.7) and OS X Mountain Lion (10.8) doesn't automatically
     # select the right stdlib.
    if(${MACOSX_VERSION} LESS 9)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}
-stdlib=libc++")
      set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}
-stdlib=libc++")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
     endif()
  endif()

  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lm")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -lm")
  set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -lm")
endif()

# Debugging CFLAGS.  Turn optimizations off; turn debugging symbols on.
if(DEBUG)
  add_definitions(-DDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -ftemplate-backtrace-limit=0")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -O0")
  # mlpack uses it's own mlpack::backtrace class based on Binary File Descriptor
  # <bfd.h> and linux Dynamic Loader <libdl.h> and more portable version in
  # future
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Bfd)
    find_package(LibDL)
    if(LIBBFD_FOUND AND LIBDL_FOUND)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
      include_directories(${LIBBFD_INCLUDE_DIRS})
      include_directories(${LIBDL_INCLUDE_DIRS})
      add_definitions(-DHAS_BFD_DL)
    else()
      message(WARNING "No libBFD and/or libDL has been found!")
    endif()
  endif()
else()
  add_definitions(-DARMA_NO_DEBUG)
  add_definitions(-DNDEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3")
endif()

# Profiling CFLAGS.  Turn profiling information on.
if(CMAKE_COMPILER_IS_GNUCC AND PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

# Find dependencies.  This isn't very robust.
find_package(Armadillo 4.100.0 REQUIRED)

# If Armadillo was compiled without ARMA_64BIT_WORD and we are on a 64-bit
# system (where size_t will be 64 bits), suggest to the user that they should
# compile Armadillo with 64-bit words.  Note that with Armadillo 5.000.0 and
# newer, ARMA_64BIT_WORD is enabled by default.
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  # Check the version, to see if ARMA_64BIT_WORD is enabled by default.
  set(ARMA_HAS_64BIT_WORD 0)
  if(NOT (${ARMADILLO_VERSION_MAJOR} LESS 5))
    set(ARMA_HAS_64BIT_WORD 1)
  else()
    # Can we open the configuration file?  If not, issue a warning.
    if(NOT EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
      message(WARNING "Armadillo configuration file "
          "(${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp) does not
exist!")
    else()
      # We are on a 64-bit system.  Does Armadillo have ARMA_64BIT_WORD enabled?
      file(READ "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp"
ARMA_CONFIG)
      string(REGEX MATCH
          "[\r\n][ ]*#define ARMA_64BIT_WORD"
          ARMA_HAS_64BIT_WORD_PRE
          "${ARMA_CONFIG}")

      string(LENGTH "${ARMA_HAS_64BIT_WORD_PRE}" ARMA_HAS_64BIT_WORD)
    endif()
  endif()

  if(ARMA_HAS_64BIT_WORD EQUAL 0)
    message(WARNING "This is a 64-bit system, but Armadillo was compiled "
        "without 64-bit index support.  Consider recompiling Armadillo with "
        "ARMA_64BIT_WORD to enable 64-bit indices (large matrix support). "
        "mlpack will still work without ARMA_64BIT_WORD defined, but will not "
        "scale to matrices with more than 4 billion elements.")
  endif()
else()
  # If we are on a 32-bit system, we must manually specify the size of the word
  # to be 32 bits, since otherwise Armadillo will produce a warning that it is
  # disabling 64-bit support.
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    add_definitions(-DARMA_32BIT_WORD)
  endif ()
endif()


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
  endif ()

  find_library(BLAS_LIBRARY
      NAMES blas libblas blas_win32_MT blas_win32
      PATHS "C:/Program Files/Armadillo"
      PATH_SUFFIXES "examples/lib_win32/")

  if (NOT BLAS_LIBRARY)
    message(FATAL_ERROR "Cannot find BLAS library (.lib)!")
  endif ()

  # Piggyback LAPACK and BLAS linking into Armadillo link.
  set(ARMADILLO_LIBRARIES
       ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARY} ${LAPACK_LIBRARY})
endif ()

# Include directories for the previous dependencies.
include_directories(${ARMADILLO_INCLUDE_DIRS})

# Unfortunately this configuration variable is necessary and will need to be
# updated as time goes on and new versions are released.
set(Boost_ADDITIONAL_VERSIONS
  "1.49.0" "1.50.0" "1.51.0" "1.52.0" "1.53.0" "1.54.0" "1.55.0")
find_package(Boost 1.49
    COMPONENTS
      program_options
      unit_test_framework
      serialization
    REQUIRED
)
include_directories(${Boost_INCLUDE_DIRS})

link_directories(${Boost_LIBRARY_DIRS})

# In Visual Studio, automatic linking is performed, so we don't need to worry
# about it.  Clear the list of libraries to link against and let Visual Studio
# handle it.
if (MSVC)
  link_directories(${Boost_LIBRARY_DIRS})
  set(Boost_LIBRARIES "")
endif ()

# For Boost testing framework (will have no effect on non-testing executables).
# This specifies to Boost that we are dynamically linking to the Boost test
# library.
add_definitions(-DBOOST_TEST_DYN_LINK)

# On Windows, things end up under Debug/ or Release/.
if (WIN32)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
else ()
  # If not on Windows, put them under more standard UNIX-like places.  This is
  # necessary, otherwise they would all end up in
  # ${CMAKE_BINARY_DIR}/src/mlpack/methods/... or somewhere else random like
  # that.
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
endif ()

# Find the mlpack library and include directory.
set(MLPACK_LIBRARY "")
set(MLPACK_INCLUDE_DIR "")
find_library(MLPACK_LIBRARY
  NAMES mlpack
  PATHS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib
)

find_path(MLPACK_INCLUDE_DIR mlpack/core.hpp
  /usr/include/
  /usr/local/include/
)

if (MLPACK_LIBRARY AND MLPACK_INCLUDE_DIR)
  mark_as_advanced(MLPACK_LIBRARY MLPACK_INCLUDE_DIR)
  include_directories(${MLPACK_INCLUDE_DIR})
else ()
  message(FATAL_ERROR "Could not find mlpack; try specifying MLPACK_LIBRARY and"
      " MLPACK_INCLUDE_DIR")
endif ()

# Finally!  Definitions of the files we are building.
add_executable(qdafn
  qdafn_main.cpp
  qdafn.hpp
  qdafn_impl.hpp
)
target_link_libraries(qdafn
  ${MLPACK_LIBRARY}
  ${Boost_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
)

add_executable(qdafn_test
  qdafn_test.cpp
)
target_link_libraries(qdafn_test
  ${MLPACK_LIBRARY}
  ${Boost_LIBRARIES}
  ${ARMADILLO_LIBRARIES}
)
