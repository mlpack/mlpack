# Build rules for the MATLAB bindings for MLPACK.  These may not work well on
# non-Linux systems.

# We need the mex compiler for this to work.
find_package(MatlabMex)

if(NOT ${MATLABMEX_FOUND})
  message(STATUS
      "MATLAB mex compiler not found; MATLAB bindings will not be built.")
  message(STATUS "If MATLAB is installed, try specifying MATLAB_ROOT.")
  return()
endif(NOT ${MATLABMEX_FOUND})

# Ignore the fact that we are setting CMAKE_SHARED_LIBRARY_CXX_FLAGS on CMake
# 2.8.9 and newer.  Because we are requiring at least CMake 2.8.5, we only have
# to check the patch version.
if(${CMAKE_PATCH_VERSION} GREATER 8)
  cmake_policy(SET CMP0018 OLD)
endif(${CMAKE_PATCH_VERSION} GREATER 8)

# Use the mex compiler to compile.
set(CMAKE_CXX_COMPILER "${MATLAB_MEX}")

# Set flags for the mex compiler, because a lot of the default CMake flags
# aren't accepted by mex.  The user who wants to customize these things should
# probably modify their mexopts.sh so that mex uses those flags by default.
# There is no easy way to tell mex to compile with profiling symbols, so that is
# not done even if PROFILE is set.
if(DEBUG)
  set(CMAKE_CXX_FLAGS "-g")
  set(CMAKE_C_FLAGS "-g")
else(DEBUG)
  set(CMAKE_CXX_FLAGS "-O")
  set(CMAKE_C_FLAGS "-O")
endif(DEBUG)

# Don't give -fPIC; mex will do that for us.
set(CMAKE_SHARED_LIBRARY_C_FLAGS "")
set(CMAKE_SHARED_LIBRARY_CXX_FLAGS "")

# Don't make 'lib<method>.mexglx'.
set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_SHARED_MODULE_PREFIX "")

# Set custom commands for mex compilation, because the flags are (in general)
# odd and different.
set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> -outdir <OBJECT_DIR> <FLAGS> -c <SOURCE>")
set(CMAKE_CXX_CREATE_SHARED_MODULE "<CMAKE_CXX_COMPILER> -cxx <LINK_FLAGS> -output <TARGET> <OBJECTS> <LINK_LIBRARIES>")
set(CMAKE_CXX_CREATE_SHARED_LIBRARY "${CMAKE_CXX_CREATE_SHARED_MODULE}")

# mex is weird because it doesn't respect the -o option, but in general it
# appears to turn <source>.cpp into <source>.o, so CMake needs to know to
# replace the extension.
set(CMAKE_CXX_OUTPUT_EXTENSION_REPLACE 1)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexa64")
  set(CMAKE_SHARED_MODULE_SUFFIX  ".mexa64")
elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86" OR ${CMAKE_SYSTEM_PROCESSOR}
    STREQUAL "i686")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".mexglx")
  set(CMAKE_SHARED_MODULE_SUFFIX  ".mexglx")
endif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64")

# Place MATLAB bindings in matlab/.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/matlab/)

include_directories(${CMAKE_SOURCE_DIR}/src/) # So we can include <mlpack/...>.

# Set MATLAB toolbox install directory.
set(MATLAB_TOOLBOX_DIR "${MATLAB_ROOT}/toolbox")

add_subdirectory(emst)
