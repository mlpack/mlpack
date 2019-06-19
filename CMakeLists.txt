cmake_minimum_required(VERSION 3.3.2)
project(mlpack C CXX)

include(CMake/cotire.cmake)

# First, define all the compilation options.
# We default to debugging mode for developers.
option(DEBUG "Compile with debugging information." OFF)
option(PROFILE "Compile with profiling information." OFF)
option(ARMA_EXTRA_DEBUG "Compile with extra Armadillo debugging symbols." OFF)
option(MATLAB_BINDINGS "Compile MATLAB bindings if MATLAB is found." OFF)
option(TEST_VERBOSE "Run test cases with verbose output." OFF)
option(BUILD_TESTS "Build tests." ON)
option(BUILD_CLI_EXECUTABLES "Build command-line executables." ON)
option(DOWNLOAD_ENSMALLEN "If ensmallen is not found, download it." ON)
option(DOWNLOAD_STB_IMAGE "Download stb_image for image loading." ON)

# Currently Python bindings aren't known to build successfully on Windows, so
# set BUILD_PYTHON_BINDINGS to OFF when the platform is Windows.
if (WIN32)
  option(BUILD_PYTHON_BINDINGS "Build Python bindings." OFF)
  option(BUILD_SHARED_LIBS
      "Compile shared libraries (if OFF, static libraries are compiled)." OFF)
  message(WARNING "By default Python bindings are not compiled for Windows because they are not known to work.  Set BUILD_PYTHON_BINDINGS to ON if you want them built.")
else ()
  option(BUILD_PYTHON_BINDINGS "Build Python bindings." ON)
  option(BUILD_SHARED_LIBS
      "Compile shared libraries (if OFF, static libraries are compiled)." ON)
endif()

# Build Markdown bindings for documentation.  This is used as part of website
# generation.
option(BUILD_MARKDOWN_BINDINGS "Build Markdown bindings for website documentation." OFF)

option(BUILD_WITH_COVERAGE
    "Build with support for code coverage tools (gcc only)." OFF)
option(MATHJAX
    "Use MathJax for HTML Doxygen output (disabled by default)." OFF)
option(FORCE_CXX11
    "Don't check that the compiler supports C++11, just assume it.  Make sure to specify any necessary flag to enable C++11 as part of CXXFLAGS." OFF)
option(USE_OPENMP "If available, use OpenMP for parallelization." ON)
enable_testing()

# Set required standard to C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include modules in the CMake directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# If we are on a Unix-like system, use the GNU install directories module.
# Otherwise set the values manually.
if (UNIX)
  include(GNUInstallDirs)
else ()
  set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
  set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
  set(CMAKE_INSTALL_MANDIR ${CMAKE_INSTALL_PREFIX}/man)
  set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_PREFIX}/share/doc/mlpack)
  set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)
endif ()

# This is as of yet unused.
#option(PGO "Use profile-guided optimization if not a debug build" ON)

# Set the CFLAGS and CXXFLAGS depending on the options the user specified.
# Only GCC-like compilers support -Wextra, and other compilers give tons of
# output for -Wall, so only -Wall and -Wextra on GCC.
if(CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Ensure that we can't compile with clang 3.4, since this causes strange
  # issues.
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
    message(FATAL_ERROR "mlpack does not build correctly with clang < 3.5.  "
        "Please upgrade your compiler and reconfigure mlpack.")
  endif ()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ftemplate-depth=1000")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
endif()

# These support libraries are used if we need to link against something
# specific.  This list is a subset of MLPACK_LIBRARIES.
set(COMPILER_SUPPORT_LIBRARIES "")

# If we are using MSVC, we need /bigobj.
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif ()

# If we are using MINGW, we need sections and big-obj, otherwise we create too
# many sections.
if(CMAKE_COMPILER_IS_GNUCC AND WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections -Wa,-mbig-obj")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections -Wa,-mbig-obj")
endif()

# If using clang, we have to link against libc++ depending on the
# OS (at least on some systems). Further, gcc sometimes optimizes calls to
# math.h functions, making -lm unnecessary with gcc, but it may still be
# necessary with clang.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (APPLE)
    # Detect OS X version. Use '/usr/bin/sw_vers -productVersion' to
    # extract V from '10.V.x'.
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
      set(CMAKE_SHARED_LINKER_FLAGS
          "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_MODULE_LINKER_FLAGS
          "${CMAKE_MODULE_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
     endif()
  endif()

  # Link everything with -lm.
  set(COMPILER_SUPPORT_LIBRARIES ${COMPILER_SUPPORT_LIBRARIES} "m")
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} "m")
  # Use -pthread, but not on OS X.
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  endif ()
endif()

# If we're using gcc, then we need to link against pthreads to use std::thread,
# which we do in the tests.
if(CMAKE_COMPILER_IS_GNUCC)
  find_package(Threads)
  set(COMPILER_SUPPORT_LIBRARIES ${COMPILER_SUPPORT_LIBRARIES}
      ${CMAKE_THREAD_LIBS_INIT})
endif()

# Setup build for test coverage
if(BUILD_WITH_COVERAGE)
  # Currently coverage only works with GNU g++
  if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # Find gcov and lcov
    find_program(GCOV gcov)
    find_program(LCOV lcov)

    if(NOT GCOV)
      message(FATAL_ERROR
          "gcov not found! gcov is required when BUILD_WITH_COVERAGE=ON.")
    endif()

    set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} "supc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fno-inline -fno-inline-small-functions -fno-default-inline -fprofile-arcs -fkeep-inline-functions")
    message(STATUS "Adding debug compile options for code coverage.")
    # Remove optimizations for better line coverage
    set(DEBUG ON)

    if(LCOV)
      configure_file(CMake/mlpack_coverage.in mlpack_coverage @ONLY)
      add_custom_target(mlpack_coverage DEPENDS mlpack_test COMMAND ${PROJECT_BINARY_DIR}/mlpack_coverage)
    else()
      message(WARNING "'lcov' not found; local coverage report is disabled. "
          "Install 'lcov' and rerun cmake to generate local coverage report.")
    endif()
  else()
    message(FATAL_ERROR "BUILD_WITH_COVERAGE can only work with GNU environment.")
  endif()
endif()

# Debugging CFLAGS.  Turn optimizations off; turn debugging symbols on.
if(DEBUG)
  if (NOT MSVC)
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -ftemplate-backtrace-limit=0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -O0")
  endif()

  # mlpack uses it's own mlpack::backtrace class based on Binary File Descriptor
  # <bfd.h> and linux Dynamic Loader <libdl.h> and more portable version in future
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Bfd)
    find_package(LibDL)
    if(LIBBFD_FOUND AND LIBDL_FOUND)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
      set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${LIBBFD_INCLUDE_DIRS}
          ${LIBDL_INCLUDE_DIRS})
      set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${LIBBFD_LIBRARIES}
          ${LIBDL_LIBRARIES})
      add_definitions(-DHAS_BFD_DL)
    else()
      message(WARNING "No libBFD and/or libDL has been found!")
    endif()
  endif()
else()
  add_definitions(-DARMA_NO_DEBUG)
  add_definitions(-DNDEBUG)
  if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -O3")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /O3")
  endif ()
endif()

# Profiling CFLAGS.  Turn profiling information on.
if(CMAKE_COMPILER_IS_GNUCC AND PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

# If the user asked for running test cases with verbose output, turn that on.
if(TEST_VERBOSE)
  add_definitions(-DTEST_VERBOSE)
endif()

# If the user asked for extra Armadillo debugging output, turn that on.
if(ARMA_EXTRA_DEBUG)
  add_definitions(-DARMA_EXTRA_DEBUG)
endif()

# Now, find the libraries we need to compile against.  Several variables can be
# set to manually specify the directory in which each of these libraries
# resides.
#   ARMADILLO_LIBRARY - location of libarmadillo.so / armadillo.lib
#   ARMADILLO_INCLUDE_DIR - directory containing <armadillo>
#   ARMADILLO_INCLUDE_DIRS - directories necessary for Armadillo includes
#   BOOST_ROOT - root of Boost installation
#   BOOST_INCLUDEDIR - include directory for Boost
#   BOOST_LIBRARYDIR - library directory for Boost
#   ENSMALLEN_INCLUDE_DIR - include directory for ensmallen
#   MATHJAX_ROOT - root of MathJax installation

find_package(Armadillo 6.500.0 REQUIRED)

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
          "(${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp) does not exist!")
    else()
      # We are on a 64-bit system.  Does Armadillo have ARMA_64BIT_WORD enabled?
      file(READ "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp" ARMA_CONFIG)
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
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})

# Find stb_image.h and stb_image_write.h.
find_package(StbImage)
# Download stb_image for image loading.
if (NOT STB_IMAGE_FOUND)
  if (DOWNLOAD_STB_IMAGE)
    set(STB_DIR "stb")
    install(DIRECTORY DESTINATION ${CMAKE_BINARY_DIR}/deps/${STB_DIR})
    file(DOWNLOAD http://mlpack.org/files/stb/stb_image.h
        "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image.h"
        STATUS STB_IMAGE_DOWNLOAD_STATUS_LIST LOG STB_IMAGE_DOWNLOAD_LOG
        SHOW_PROGRESS)
    list(GET STB_IMAGE_DOWNLOAD_STATUS_LIST 0 STB_IMAGE_DOWNLOAD_STATUS)
    file(DOWNLOAD http://mlpack.org/files/stb/stb_image_write.h
        "${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image_write.h"
        STATUS STB_IMAGE_WRITE_DOWNLOAD_STATUS_LIST
        LOG STB_IMAGE_WRITE_DOWNLOAD_LOG
        SHOW_PROGRESS)
    list(GET STB_IMAGE_WRITE_DOWNLOAD_STATUS_LIST 0
        STB_IMAGE_WRITE_DOWNLOAD_STATUS)
    if (STB_IMAGE_DOWNLOAD_STATUS EQUAL 0 AND
        STB_IMAGE_WRITE_DOWNLOAD_STATUS EQUAL 0)
        set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
            "${CMAKE_BINARY_DIR}/deps/${STB_DIR}")
        message(STATUS
            "Successfully downloaded stb into ${CMAKE_BINARY_DIR}/deps/${STB_DIR}/")
        # Now we have to also ensure these header files get installed.
        install(FILES ${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        install(FILES ${CMAKE_BINARY_DIR}/deps/${STB_DIR}/stb_image_write.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        add_definitions(-DHAS_STB)
    else ()
      list(GET STB_IMAGE_DOWNLOAD_STATUS_LIST 1 STB_DOWNLOAD_ERROR)
      message(WARNING
          "Could not download stb! Error code ${STB_DOWNLOAD_STATUS}: ${STB_DOWNLOAD_ERROR}!  Error log: ${STB_DOWNLOAD_LOG}")
    endif ()
  else ()
    message(WARNING
        "stb/stb_image.h is not installed. Image utilites will not be available!")
  endif ()
else ()
  # Already has STB installed.
  add_definitions(-DHAS_STB)
endif ()

# Find ensmallen.
# Once ensmallen is readily available in package repos, the automatic downloader
# here can be removed.
find_package(Ensmallen 1.10.0)
if (NOT ENSMALLEN_FOUND)
  if (DOWNLOAD_ENSMALLEN)
    file(DOWNLOAD http://www.ensmallen.org/files/ensmallen-latest.tar.gz
        "${CMAKE_BINARY_DIR}/deps/ensmallen-latest.tar.gz"
        STATUS ENS_DOWNLOAD_STATUS_LIST LOG ENS_DOWBLOAD_LOG
        SHOW_PROGRESS)
    list(GET ENS_DOWNLOAD_STATUS_LIST 0 ENS_DOWNLOAD_STATUS)
    if (ENS_DOWNLOAD_STATUS EQUAL 0)
      execute_process(COMMAND ${CMAKE_COMMAND} -E
          tar xzf "${CMAKE_BINARY_DIR}/deps/ensmallen-latest.tar.gz"
          WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/deps/")

      # Get the name of the directory.
      file (GLOB ENS_DIRECTORIES RELATIVE "${CMAKE_BINARY_DIR}/deps/"
          "${CMAKE_BINARY_DIR}/deps/ensmallen-[0-9]*.[0-9]*.[0-9]*")
      list(LENGTH ENS_DIRECTORIES ENS_DIRECTORIES_LEN)
      if (ENS_DIRECTORIES_LEN EQUAL 1)
        list(GET ENS_DIRECTORIES 0 ENSMALLEN_INCLUDE_DIR)
        set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS}
            "${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include")
        message(STATUS
            "Successfully downloaded ensmallen into ${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/")

        # Now we have to also ensure these header files get installed.
        install(DIRECTORY ${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include/ensmallen_bits/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/ensmallen_bits)
        install(FILES ${CMAKE_BINARY_DIR}/deps/${ENSMALLEN_INCLUDE_DIR}/include/ensmallen.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
      else ()
        message(FATAL_ERROR "Problem unpacking ensmallen!  Expected only one directory ensmallen-x.y.z/; found ${ENS_DIRECTORIES}. Try removing the directory ${CMAKE_BINARY_DIR}/deps and reconfiguring.")
      endif ()
    else ()
      list(GET ENS_DOWNLOAD_STATUS_LIST 1 ENS_DOWNLOAD_ERROR)
      message(FATAL_ERROR
          "Could not download ensmallen! Error code ${ENS_DOWNLOAD_STATUS}: ${ENS_DOWNLOAD_ERROR}!  Error log: ${ENS_DOWNLOAD_LOG}")
    endif ()
  else ()
    # Release versions will have ensmallen packaged with the release so we can
    # just reference that.
    if (EXISTS "${CMAKE_SOURCE_DIR}/src/mlpack/core/optimizers/ensmallen/ensmallen.hpp")
      set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS}
          "${CMAKE_SOURCE_DIR}/src/mlpack/core/optimizers/ensmallen")
    else ()
      message(FATAL_ERROR
          "Cannot find ensmallen headers!  Try setting ENSMALLEN_INCLUDE_DIR!")
    endif ()
  endif ()
else ()
  set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${ENSMALLEN_INCLUDE_DIR}")
endif ()

# Unfortunately this configuration variable is necessary and will need to be
# updated as time goes on and new versions are released.
set(Boost_ADDITIONAL_VERSIONS
  "1.69.0" "1.69"
  "1.68.0" "1.68"
  "1.67.0" "1.67"
  "1.66.0" "1.66"
  "1.65.1" "1.65.0" "1.65"
  "1.64.1" "1.64.0" "1.64"
  "1.63.1" "1.63.0" "1.63"
  "1.62.1" "1.62.0" "1.62"
  "1.61.1" "1.61.0" "1.61"
  "1.60.1" "1.60.0" "1.60"
  "1.59.1" "1.59.0" "1.59"
  "1.58.1" "1.58.0" "1.58"
  "1.57.1" "1.57.0" "1.57"
  "1.56.1" "1.56.0" "1.56"
  "1.55.1" "1.55.0" "1.55"
  "1.54.1" "1.54.0" "1.54"
  "1.53.1" "1.53.0" "1.53"
  "1.52.1" "1.52.0" "1.52"
  "1.51.1" "1.51.0" "1.51"
  "1.50.1" "1.50.0" "1.50"
  "1.49.1" "1.49.0" "1.49")
find_package(Boost 1.49
    COMPONENTS
      program_options
      unit_test_framework
      serialization
    REQUIRED
)

link_directories(${Boost_LIBRARY_DIRS})

# In Visual Studio, automatic linking is performed, so we don't need to worry
# about it.  Clear the list of libraries to link against and let Visual Studio
# handle it.
if (MSVC)
  link_directories(${Boost_LIBRARY_DIRS})
  set(Boost_LIBRARIES "")
endif ()

set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${Boost_LIBRARIES})
set(MLPACK_LIBRARY_DIRS ${MLPACK_LIBRARY_DIRS} ${Boost_LIBRARY_DIRS})

# For Boost testing framework (will have no effect on non-testing executables).
# This specifies to Boost that we are dynamically linking to the Boost test
# library.
add_definitions(-DBOOST_TEST_DYN_LINK)

# Detect OpenMP support in a compiler. If the compiler supports OpenMP, flags
# to compile with OpenMP are returned and added and the HAS_OPENMP definition
# is added for compilation.
#
# This way we can skip calls to functions defined in omp.h with code like:
# #ifdef HAS_OPENMP
# {
#   ... openMP code here ...
# }
# #endif
if (USE_OPENMP)
  find_package(OpenMP)
endif ()

if (OPENMP_FOUND)
  add_definitions(-DHAS_OPENMP)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
  # Disable warnings for all the unknown OpenMP pragmas.
  if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
  else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4068")
  endif ()
  set(OpenMP_CXX_FLAGS "")
endif ()

# Create a 'distclean' target in case the user is using an in-source build for
# some reason.
include(CMake/TargetDistclean.cmake OPTIONAL)

include_directories(BEFORE ${MLPACK_INCLUDE_DIRS})
include_directories(BEFORE ${CMAKE_SOURCE_DIR}/src/)

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

# Determine whether or not this is a git repository, so that we can set the
# version number if necessary.
find_package(Git)
set (USING_GIT "NO")
if (GIT_FOUND)
  # Run 'git rev-parse HEAD' to find out if this is a working copy. If the
  # return code is not 0, then it isn't.
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE MLPACK_TMP_REV_INFO
      ERROR_VARIABLE MLPACK_TMP_REV_INFO_ERROR
      RESULT_VARIABLE MLPACK_TMP_REV_INFO_RESULT
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  if (${MLPACK_TMP_REV_INFO_RESULT} EQUAL 0)
    set (USING_GIT "YES")
    add_definitions(-DMLPACK_GIT_VERSION)
    include(CMake/CreateGitVersionHeader.cmake)

    add_custom_target(mlpack_gitversion ALL
        COMMAND ${CMAKE_COMMAND} -P CMake/CreateGitVersionHeader.cmake
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Updating gitversion.hpp (if necessary)")
  # Add gitversion.hpp to the list of sources.
  set(MLPACK_SRCS ${MLPACK_SRCS}
      "${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/gitversion.hpp")
  endif ()
endif ()

# Create a target to generate arma_config.hpp, which is used to warn the user
# when they are doing something stupid when linking something against mlpack.
include(CMake/CreateArmaConfigInfo.cmake)

add_custom_target(mlpack_arma_config ALL
    COMMAND ${CMAKE_COMMAND}
        -D ARMADILLO_INCLUDE_DIR="${ARMADILLO_INCLUDE_DIR}"
        -D ARMADILLO_VERSION_MAJOR="${ARMADILLO_VERSION_MAJOR}"
        -D OPENMP_FOUND="${OPENMP_FOUND}"
        -P CMake/CreateArmaConfigInfo.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Updating arma_config.hpp (if necessary)")
set(MLPACK_SRCS ${MLPACK_SRCS}
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mlpack/core/util/arma_config.hpp")

# Make a target to generate the man page documentation, but only if we are on a
# UNIX-like system.
if (BUILD_CLI_EXECUTABLES AND UNIX)
  find_program(TXT2MAN txt2man)

  # It's not a requirement that we make man pages.
  if (NOT TXT2MAN)
    message(WARNING "txt2man not found; man pages will not be generated.")
  else ()
    # We have the tools.  We can make them.
    add_custom_target(man ALL
        ${CMAKE_CURRENT_SOURCE_DIR}/CMake/allexec2man.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/CMake/exec2man.sh
            ${CMAKE_BINARY_DIR}/share/man
        WORKING_DIRECTORY
          ${CMAKE_BINARY_DIR}/bin
        COMMENT "Generating man pages from built executables."
    )

    # Set the rules to install the documentation.
    install(DIRECTORY ${CMAKE_BINARY_DIR}/share/man/
        DESTINATION ${CMAKE_INSTALL_MANDIR})
  endif ()
endif ()

# Recurse into the rest of the project.
add_subdirectory(src/mlpack)

# If we need to keep gitversion.hpp up to date, then make sure the mlpack target
# depends on it.
if (USING_GIT STREQUAL "YES")
  add_dependencies(mlpack_headers mlpack_gitversion)
endif ()

# Make the mlpack_arma_config target depend on mlpack (we couldn't do this
# before the add_subdirectory() call because the mlpack target didn't exist
# before that).
add_dependencies(mlpack_headers mlpack_arma_config)

# Make a target to generate the documentation.  If Doxygen isn't installed, then
# I guess this option will just be unavailable.
find_package(Doxygen)
if (DOXYGEN_FOUND)
  if (MATHJAX)
    find_package(MathJax)
    if (NOT MATHJAX_FOUND)
      message(STATUS "Using MathJax at the MathJax Content Delivery Network. "
          "Be careful, formulas will not be shown without the internet.")
    endif ()
  endif ()
  # Preprocess the Doxyfile.  This is done before 'make doc'.
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/Doxyfile
      PRE_BUILD
      COMMAND ${CMAKE_COMMAND}
          -D DESTDIR=${CMAKE_BINARY_DIR}
          -D MATHJAX="${MATHJAX}"
          -D MATHJAX_FOUND="${MATHJAX_FOUND}"
          -D MATHJAX_PATH="${MATHJAX_PATH}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/CMake/GenerateDoxyfile.cmake"
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
      COMMENT "Creating Doxyfile to generate Doxygen documentation"
  )

  # Generate documentation.
  add_custom_target(doc
      COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_BINARY_DIR}/Doxyfile"
      DEPENDS "${CMAKE_BINARY_DIR}/Doxyfile"
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      COMMENT "Generating API documentation with Doxygen"
  )

  install(DIRECTORY ${CMAKE_BINARY_DIR}/doc/html
      DESTINATION ${CMAKE_INSTALL_DOCDIR}
      COMPONENT doc
      OPTIONAL
  )
endif ()

# Create the pkg-config file, if we have pkg-config.
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
  # mlpack.pc must be generated as a separate target, otherwise it is possible
  # that the given version could be out of date.  We don't need to worry about
  # the library or include directories changing, because CMake will re-run this
  # portion of the code whenever any of those changes.  But the version must be
  # re-extracted every time the library is built.

  # So, we have to parse our list of library directories, libraries, and include
  # directories in order to get the correct line to give to pkg-config.
  # Next, adapt the list of include directories.
  list(REMOVE_DUPLICATES MLPACK_INCLUDE_DIRS)
  foreach (incldir ${MLPACK_INCLUDE_DIRS})
    # Filter out some obviously unnecessary directories.
    if (NOT "${incldir}" STREQUAL "/usr/include")
      set(MLPACK_INCLUDE_DIRS_STRING
          "${MLPACK_INCLUDE_DIRS_STRING} -I${incldir}")
    endif ()
  endforeach ()
  # Add the install directory too.
  set(MLPACK_INCLUDE_DIRS_STRING
      "${MLPACK_INCLUDE_DIRS_STRING} -I${CMAKE_INSTALL_PREFIX}/include/")

  # Create the list of link directories.
  set(MLPACK_LIBRARIES_LIST)
  foreach (linkdir ${MLPACK_LIBRARY_DIRS})
    list(APPEND MLPACK_LIBRARIES_LIST "-L${linkdir}")
  endforeach ()

  foreach(lib ${MLPACK_LIBRARIES})
    string(SUBSTRING "${lib}" 0 1 first)
    if ("${first}" STREQUAL "/")
      # We need to split the directory and the library.
      string(REGEX REPLACE "(.*/)[^/]*$" "\\1" library_dir "${lib}")
      string(REGEX REPLACE ".*/lib([^/]*).so.*$" "\\1" library_name "${lib}")

      list(APPEND MLPACK_LIBRARIES_LIST "-L${library_dir}")
      list(APPEND MLPACK_LIBRARIES_LIST "-l${library_name}")
    else ()
      list(APPEND MLPACK_LIBRARIES_LIST "-l${lib}")
    endif ()
  endforeach ()
  # Don't forget to add mlpack as a dependency too.
  list(APPEND MLPACK_LIBRARIES_LIST "-L${CMAKE_INSTALL_PREFIX}/lib/")
  list(APPEND MLPACK_LIBRARIES_LIST "-lmlpack")

  # Filter duplicate dependencies and directories.
  list(REMOVE_DUPLICATES MLPACK_LIBRARIES_LIST)

  # Filter out known unnecessary directories.
  list(REMOVE_ITEM MLPACK_LIBRARIES_LIST
      "-L/usr/lib"
      "-L/usr/lib/"
      "-L/usr/lib/x86_64-linux-gnu"
      "-L/usr/lib/x86_64-linux-gnu/"
      "-L/usr/lib/i386-linux-gnu"
      "-L/usr/lib/i386-linux-gnu/")

  string(REPLACE ";" " " MLPACK_LIBRARIES_STRING "${MLPACK_LIBRARIES_LIST}")

  # Do first stage of configuration.
  set(MLPACK_VERSION_STRING "@MLPACK_VERSION_STRING@")
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/CMake/mlpack.pc.in
    ${CMAKE_BINARY_DIR}/CMake/mlpack.pc.in.partial @ONLY)

  add_custom_target(pkgconfig ALL
      ${CMAKE_COMMAND}
          -P "${CMAKE_CURRENT_SOURCE_DIR}/CMake/GeneratePkgConfig.cmake"
      DEPENDS mlpack_headers
      COMMENT "Generating mlpack.pc (pkg-config) file.")

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/mlpack.pc
      DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig/)
endif ()
