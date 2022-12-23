cmake_minimum_required(VERSION 3.6)
project(mlpack C CXX)

include(CMake/cotire.cmake)
include(CMake/CheckHash.cmake)
include(CMake/Autodownload.cmake)
include(CMake/ConfigureCrossCompile.cmake)
include(CMake/CheckAtomic.cmake)

# First, define all the compilation options.
# We default to debugging mode for developers.
option(DEBUG "Compile with debugging information." OFF)
option(PROFILE "Compile with profiling information." OFF)
option(ARMA_EXTRA_DEBUG "Compile with extra Armadillo debugging symbols." OFF)
option(TEST_VERBOSE "Run test cases with verbose output." OFF)
option(BUILD_TESTS "Build tests. (Note: time consuming!)" OFF)
option(BUILD_CLI_EXECUTABLES "Build command-line executables." ON)
option(DOWNLOAD_DEPENDENCIES "Automatically download dependencies if not available." OFF)
option(BUILD_GO_SHLIB "Build Go shared library." OFF)

# Set minimum library versions required by mlpack.
#
# For Armadillo, try to keep the minimum required version less than or equal to
# what's available on the current Ubuntu LTS or most recent stable RHEL release.
# See https://github.com/mlpack/mlpack/issues/3033 for some more discussion.
set(ARMADILLO_VERSION "9.800")
set(ENSMALLEN_VERSION "2.10.0")
set(CEREAL_VERSION "1.1.2")

# If BUILD_SHARED_LIBS is OFF then the mlpack library will be built statically.
# In addition, all mlpack CLI bindings will be linked statically as well.
if (WIN32)
  option(BUILD_SHARED_LIBS
      "Compile shared objects for tests and bindings (if OFF, static libraries and binaries are compiled)." OFF)

  set(DLL_COPY_DIRS "" CACHE STRING "List of directories (separated by ';') containing DLLs to copy for runtime.")
  set(DLL_COPY_LIBS "" CACHE STRING "List of DLLs (separated by ';') that should be copied for runtime.")
elseif(CMAKE_CROSSCOMPILING)
  option(BUILD_SHARED_LIBS
      "Compile shared libraries (if OFF, static libraries and binaries are compiled)." OFF)
else()
  option(BUILD_SHARED_LIBS
      "Compile shared objects for tests and bindings (if OFF, static libraries and binaries are compiled)." ON)
endif()

# Enable auto-download if we are cross compiling.
if (CMAKE_CROSSCOMPILING)
  set(DOWNLOAD_DEPENDENCIES ON)
endif()

# Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES.
if (NOT BUILD_SHARED_LIBS)
  if(WIN32)
    list(INSERT CMAKE_FIND_LIBRARY_SUFFIXES 0 .lib .a)
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  endif()
endif()

# Detect whether the user passed BUILD_PYTHON_BINDINGS in order to determine if
# we should fail if Python isn't found.
if (BUILD_PYTHON_BINDINGS)
  set(FORCE_BUILD_PYTHON_BINDINGS ON)
else()
  set(FORCE_BUILD_PYTHON_BINDINGS OFF)
endif()
option(BUILD_PYTHON_BINDINGS "Build Python bindings." OFF)

# Detect whether the user passed BUILD_JULIA_BINDINGS in order to determine if
# we should fail if Julia isn't found.
if (BUILD_JULIA_BINDINGS)
  set(FORCE_BUILD_JULIA_BINDINGS ON)
else()
  set(FORCE_BUILD_JULIA_BINDINGS OFF)
endif()
option(BUILD_JULIA_BINDINGS "Build Julia bindings." OFF)

# Detect whether the user passed BUILD_GO_BINDINGS in order to determine if
# we should fail if Go isn't found.
if (BUILD_GO_BINDINGS)
  set(FORCE_BUILD_GO_BINDINGS ON)
else()
  set(FORCE_BUILD_GO_BINDINGS OFF)
endif()
option(BUILD_GO_BINDINGS "Build Go bindings." OFF)

# If building Go bindings then build go shared libraries.
if (BUILD_GO_BINDINGS)
  set(BUILD_GO_SHLIB ON)
endif()

# Detect whether the user passed BUILD_R_BINDINGS in order to determine if
# we should fail if R isn't found.
if (BUILD_R_BINDINGS)
  set(FORCE_BUILD_R_BINDINGS ON)
else()
  set(FORCE_BUILD_R_BINDINGS OFF)
endif()
option(BUILD_R_BINDINGS "Build R bindings." OFF)
# Build Markdown bindings for documentation.  This is used as part of website
# generation.
option(BUILD_MARKDOWN_BINDINGS "Build Markdown bindings for website documentation." OFF)

option(MATHJAX
    "Use MathJax for HTML Doxygen output (disabled by default)." OFF)
option(USE_OPENMP "If available, use OpenMP for parallelization." ON)
enable_testing()

# Set required standard to C++14.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ensure that GCC is new enough, if the compiler is GCC.
if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5)
  message(FATAL_ERROR "GCC version (${CMAKE_CXX_COMPILER_VERSION}) is too old! 5.x or newer is required.")
endif ()

# Include modules in the CMake directory.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

# If we are not using Visual Studio, use the GNU install directories module.
# Otherwise set the values manually.
if (NOT MSVC)
  include(GNUInstallDirs)
else ()
  set(CMAKE_INSTALL_BINDIR ${CMAKE_INSTALL_PREFIX}/bin)
  set(CMAKE_INSTALL_LIBDIR ${CMAKE_INSTALL_PREFIX}/lib)
  set(CMAKE_INSTALL_MANDIR ${CMAKE_INSTALL_PREFIX}/man)
  set(CMAKE_INSTALL_DOCDIR ${CMAKE_INSTALL_PREFIX}/share/doc/mlpack)
  set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_PREFIX}/include)
endif ()

# This is as of yet unused.
# option(PGO "Use profile-guided optimization if not a debug build" ON)

# Set the CFLAGS and CXXFLAGS depending on the options the user specified.
# Only GCC-like compilers support -Wextra, and other compilers give tons of
# output for -Wall, so only -Wall and -Wextra on GCC.
if (CMAKE_COMPILER_IS_GNUCC OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Ensure that we can't compile with clang 3.4, since this causes strange
  # issues.
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.5)
    message(FATAL_ERROR "mlpack does not build correctly with clang < 3.5.  "
        "Please upgrade your compiler and reconfigure mlpack.")
  endif ()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -ftemplate-depth=1000")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")

  # To remove unused functions warnings.
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
endif()

# Check if atomics need -latomic linking.
#include(CheckAtomic)
if (NOT HAVE_CXX_ATOMICS_WITHOUT_LIB AND
    NOT HAVE_CXX_ATOMICS64_WITHOUT_LIB AND
    NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -latomic")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -latomic")
endif ()

# If we are using MSVC, we need /bigobj.
if (MSVC)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj /Zm200 /Zc:__cplusplus")
endif ()

# If we are using MINGW, we need sections and big-obj, otherwise we create too
# many sections.
if (CMAKE_COMPILER_IS_GNUCC AND WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wa,-mbig-obj")
endif()

# If using clang, we have to link against libc++ depending on the
# OS (at least on some systems). Further, gcc sometimes optimizes calls to
# math.h functions, making -lm unnecessary with gcc, but it may still be
# necessary with clang.
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  if (APPLE)
    # Detect OS X version. Use '/usr/bin/sw_vers -productVersion' to
    # extract V from '10.V.x'.
    exec_program(/usr/bin/sw_vers ARGS
        -productVersion OUTPUT_VARIABLE MACOSX_VERSION_RAW)
    string(REGEX REPLACE
        "([0-9]+)(\\.([0-9]+).*)*" "\\1"
        MACOSX_MAJOR_VERSION
        "${MACOSX_VERSION_RAW}")

    string(REGEX REPLACE
        "([0-9]+)(\\.([0-9]+).*)*" "\\3"
        MACOSX_MINOR_VERSION
        "${MACOSX_VERSION_RAW}")

     # OSX Lion (10.7) and OS X Mountain Lion (10.8) doesn't automatically
     # select the right stdlib.
    if (${MACOSX_MAJOR_VERSION} LESS 11 AND ${MACOSX_MINOR_VERSION} LESS 9)
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_SHARED_LINKER_FLAGS
          "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_MODULE_LINKER_FLAGS
          "${CMAKE_MODULE_LINKER_FLAGS} -stdlib=libc++")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
     endif()
  endif()

  # Link everything with -lm.
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} "m")
  # Use -pthread, but not on OS X.
  if (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  endif ()
endif()

# If we're using gcc, then we need to link against pthreads to use std::thread,
# which we do in the tests.
if (CMAKE_COMPILER_IS_GNUCC)
  find_package(Threads)
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
endif()

# Debugging CFLAGS.  Turn optimizations off; turn debugging symbols on.
if (DEBUG)
  if (NOT MSVC)
    add_definitions(-DDEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -ftemplate-backtrace-limit=0")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -g -O0")
  endif()

  # mlpack uses it's own mlpack::backtrace class based on Binary File Descriptor
  # <bfd.h> and linux Dynamic Loader <libdl.h> and more portable version in future
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Bfd)
    find_package(LibDL)
    if (LIBBFD_FOUND AND LIBDL_FOUND)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic")
      set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${LIBBFD_INCLUDE_DIRS}
          ${LIBDL_INCLUDE_DIRS})
      set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${LIBBFD_LIBRARIES}
          ${LIBDL_LIBRARIES})
      add_definitions(-DMLPACK_HAS_BFD_DL)
    else()
      message(WARNING "No libBFD and/or libDL has been found!")
    endif()
  endif()
else()
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
if (CMAKE_COMPILER_IS_GNUCC AND PROFILE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

# If the user asked for extra Armadillo debugging output, turn that on.
if (ARMA_EXTRA_DEBUG)
  add_definitions(-DARMA_EXTRA_DEBUG)
endif()

# Now, find the libraries we need to compile against.  Several variables can be
# set to manually specify the directory in which each of these libraries
# resides.
#   ARMADILLO_LIBRARY - location of libarmadillo.so / armadillo.lib
#   ARMADILLO_INCLUDE_DIR - directory containing <armadillo>
#   ARMADILLO_INCLUDE_DIRS - directories necessary for Armadillo includes
#   CEREAL_INCLUDE_DIR - include directory for cereal
#   ENSMALLEN_INCLUDE_DIR - include directory for ensmallen
#   STB_IMAGE_INCLUDE_DIR - include directory for STB image library
#   MATHJAX_ROOT - root of MathJax installation

# Download and compile OpenBLAS if we are cross compiling mlpack for a specific
# architecture. The function takes the version of OpenBLAS as variable.
if (CMAKE_CROSSCOMPILING)
  search_openblas(0.3.13)
endif()

if (NOT DOWNLOAD_DEPENDENCIES)
  find_package(Armadillo "${ARMADILLO_VERSION}" REQUIRED)
else()
  find_package(Armadillo "${ARMADILLO_VERSION}")
  if (NOT ARMADILLO_FOUND)
    if (NOT CMAKE_CROSSCOMPILING)
      find_package(BLAS QUIET)
      find_package(LAPACK QUIET)
      if (NOT BLAS_FOUND AND NOT LAPACK_FOUND)
        message(FATAL_ERROR "Can not find BLAS or LAPACK!  These are required for Armadillo.  Please install one of them---or install Armadillo---before installing mlpack.")
      endif()
    endif()
    get_deps(http://files.mlpack.org/armadillo-10.3.0.tar.gz armadillo armadillo-10.3.0.tar.gz)
    set(ARMADILLO_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_package(Armadillo REQUIRED)
  endif()
endif()
# Include directories for the previous dependencies.
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${ARMADILLO_INCLUDE_DIRS})
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})

# Find stb_image.h and stb_image_write.h.
if (NOT DOWNLOAD_DEPENDENCIES)
  find_package(StbImage)
else()
  find_package(StbImage)
  if (NOT STB_IMAGE_FOUND)
    get_deps(http://mlpack.org/files/stb.tar.gz stb stb.tar.gz)
    set(STB_IMAGE_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_package(StbImage REQUIRED)
  endif()
endif()

if (STB_IMAGE_FOUND)
  add_definitions(-DHAS_STB)
  set(STB_AVAILABLE "1")
  set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${STB_IMAGE_INCLUDE_DIR}")

  # Make sure that we can link STB in multiple translation units.
  include(CMake/TestStaticSTB.cmake)
  if (NOT CMAKE_HAS_WORKING_STATIC_STB)
    message(FATAL_ERROR "STB implementations's static mode cannot link across "
        "multiple translation units!  Try upgrading your STB implementation, "
        "or using the auto-downloader (set DOWNLOAD_DEPENDENCIES=ON in the "
        "CMake configuration command.")
  endif ()
endif()

# Find ensmallen.
if (NOT DOWNLOAD_DEPENDENCIES)
  find_package(Ensmallen "${ENSMALLEN_VERSION}" REQUIRED)
else()
  find_package(Ensmallen "${ENSMALLEN_VERSION}")
  if (NOT ENSMALLEN_FOUND)
    get_deps(http://www.ensmallen.org/files/ensmallen-2.19.0.tar.gz ensmallen ensmallen-latest.tar.gz)
    set(ENSMALLEN_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_package(Ensmallen REQUIRED)
  endif()
endif()
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} "${ENSMALLEN_INCLUDE_DIR}")

# Find cereal.
if (NOT DOWNLOAD_DEPENDENCIES)
  find_package(cereal "${CEREAL_VERSION}" REQUIRED)
else()
  find_package(cereal "${CEREAL_VERSION}")
  if (NOT CEREAL_FOUND)
    get_deps(https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.0.tar.gz cereal cereal-1.3.0.tar.gz)
    set(CEREAL_INCLUDE_DIR ${GENERIC_INCLUDE_DIR})
    find_package(cereal REQUIRED)
  endif()
endif()
set(MLPACK_INCLUDE_DIRS ${MLPACK_INCLUDE_DIRS} ${CEREAL_INCLUDE_DIR})

# Detect OpenMP support in a compiler. If the compiler supports OpenMP, flags
# to compile with OpenMP are returned and added.  Note that MSVC does not
# support a new-enough version of OpenMP to be useful.
if (USE_OPENMP)
  find_package(OpenMP)
endif ()

if (OpenMP_FOUND AND OpenMP_CXX_VERSION VERSION_GREATER_EQUAL 3.0.0)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
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
include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src/)

# On Windows, things end up under Debug/ or Release/.
if (WIN32)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  # Copy all necessary DLLs for runtime to the build directory.
  # This is a little hackish, but I can't figure out clear ways to make CMake
  # consistently link everything 100% statically across platforms or set the
  # runtime path right always, so this is the best I know how to do for now.
  foreach(dir ${DLL_COPY_DIRS})
    file(GLOB dir_dll_list "${dir}/*.dll")
    file(COPY ${dir_dll_list} DESTINATION ${CMAKE_BINARY_DIR}/Release/)
    file(COPY ${dir_dll_list} DESTINATION ${CMAKE_BINARY_DIR}/Debug/)
  endforeach ()

  foreach(file ${DLL_COPY_LIBS})
    file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/Release/)
    file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/Debug/)
  endforeach()
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
    install(DIRECTORY "${CMAKE_BINARY_DIR}/share/man/"
        DESTINATION "${CMAKE_INSTALL_MANDIR}")
  endif ()
endif ()

# Finally, add any cross-compilation support libraries (they may need to come
# last).  If we are not cross-compiling, no changes will happen here.
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${CROSS_COMPILE_SUPPORT_LIBRARIES})

# Recurse into the rest of the project.
add_subdirectory(src/mlpack)

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
      string(REGEX REPLACE ".*/lib([^/]*)[.][a-z]*[.]*$" "\\1" library_name "${lib}")

      list(APPEND MLPACK_LIBRARIES_LIST "-L${library_dir}")
      list(APPEND MLPACK_LIBRARIES_LIST "-l${library_name}")
    elseif ("${first}" STREQUAL "-")
      # This argument is already in the right format.  (This happens with, e.g.,
      # `-lpthread`.)
      list(APPEND MLPACK_LIBRARIES_LIST "${lib}")
    else ()
      list(APPEND MLPACK_LIBRARIES_LIST "-l${lib}")
    endif ()
  endforeach ()

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
          -D MLPACK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
          -P "${CMAKE_CURRENT_SOURCE_DIR}/CMake/GeneratePkgConfig.cmake"
      COMMENT "Generating mlpack.pc (pkg-config) file.")

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/mlpack.pc"
      DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig/")

endif ()
