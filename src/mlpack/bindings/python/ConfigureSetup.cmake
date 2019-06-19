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

configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})
