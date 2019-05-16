# ConfigureSetup.cmake: generate the setup.py file given several environment
# variables.
#
# This file depends on the following variables being set:
#
# - SETUP_PY_IN: location of input file
# - SETUP_PY_OUT: location of output file
# - PACKAGE_VERSION: version of package
# - Boost_SERIALIZATION_LIBRARY: location of Boost serialization library
# - MLPACK_LIBRARY: location of mlpack library
# - MLPACK_PYXS: list of pyx files
# - OpenMP_CXX_FLAGS: OpenMP C++ compilation flags
# - DISABLE_CFLAGS: list of CFLAGS or CXXFLAGS to be disabled
# - CYTHON_INCLUDE_DIRECTORIES: include directories for Cython
# - MLPACK_LIBDIR: path to mlpack libraries
# - BOOST_LIBDIRS: paths to Boost libraries
# - OUTPUT_DIR: binary output directory for CMake

get_filename_component(Boost_SERIALIZATION_LIB_NAME_IN
    ${Boost_SERIALIZATION_LIBRARY} NAME_WE)
get_filename_component(MLPACK_LIB_NAME_IN
    ${MLPACK_LIBRARY} NAME_WE)

# Strip 'lib' off the front of the name, if it's there.
string(REGEX REPLACE "^lib" "" Boost_SERIALIZATION_LIB_NAME
    "${Boost_SERIALIZATION_LIB_NAME_IN}")
string(REGEX REPLACE "^lib" "" MLPACK_LIB_NAME "${MLPACK_LIB_NAME_IN}")

configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})
