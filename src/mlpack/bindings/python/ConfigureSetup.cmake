# ConfigureSetup.cmake: generate the setup.py file given several environment
# variables.
#
# This file depends on the following variables being set:
#
# - SETUP_PY_IN: location of input file
# - SETUP_PY_OUT: location of output file
# - PACKAGE_VERSION: version of package
# - Boost_SERIALIZATION_LIBRARY: location of Boost serialization library
# - ARMADILLO_LIBRARIES: space-separated list of Armadillo dependencies
# - MLPACK_LIBRARY: location of mlpack library
# - MLPACK_PYXS: list of pyx files
# - OpenMP_CXX_FLAGS: OpenMP C++ compilation flags
# - DISABLE_CFLAGS: list of CFLAGS or CXXFLAGS to be disabled
# - CYTHON_INCLUDE_DIRECTORIES: include directories for Cython
# - MLPACK_LIBDIR: path to mlpack libraries
# - BOOST_LIBDIRS: paths to Boost libraries
# - OUTPUT_DIR: binary output directory for CMake

# It's possible that the FindBoost CMake script may have returned a Boost
# library with "lib" improperly prepended to it.  So we need to see if the file
# exists, and if it doesn't, but it has a "lib" in it, then we will try
# stripping the "lib" off the front.
if (NOT EXISTS "$(Boost_SERIALIZATION_LIBRARY}")
  # Split the filename to see if it starts with lib.
  set(Boost_SERIALIZATION_LIBRARY_ORIG "${Boost_SERIALIZATION_LIBRARY}")
  get_filename_component(SER_LIB_DIRECTORY "${Boost_SERIALIZATION_LIBRARY}"
      DIRECTORY)
  get_filename_component(SER_LIB_FILENAME "${Boost_SERIALIZATION_LIBRARY}" NAME)

  # Strip any preceding "lib/".
  string(REGEX REPLACE "^lib" "" STRIPPED_FILENAME "${SER_LIB_FILENAME}")
  set(Boost_SERIALIZATION_LIBRARY "${SER_LIB_DIRECTORY}/${STRIPPED_FILENAME}")

  if (NOT EXISTS "${Boost_SERIALIZATION_LIBRARY}")
    # We didn't find it, so for ease of debugging just revert to the original.
    set (Boost_SERIALIZATION_LIBRARY "${Boost_SERIALIZATION_LIBRARY_ORIG}")
  endif ()
endif ()

configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})
