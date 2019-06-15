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
get_filename_component(Boost_SERIALIZATION_LIB_NAME_IN
    ${Boost_SERIALIZATION_LIBRARY} NAME_WE)
get_filename_component(MLPACK_LIB_NAME_IN
    ${MLPACK_LIBRARY} NAME_WE)

# Strip 'lib' off the front of the name, if it's there.
string(REGEX REPLACE "^lib" "" Boost_SERIALIZATION_LIB_NAME
    "${Boost_SERIALIZATION_LIB_NAME_IN}")
string(REGEX REPLACE "^lib" "" MLPACK_LIB_NAME "${MLPACK_LIB_NAME_IN}")

# There may be multiple Armadillo libraries, so first convert the string we got
# to a list.
string(REPLACE " " ";" ARMADILLO_LIBRARIES_LIST "${ARMADILLO_LIBRARIES}")
set(ARMADILLO_LIBRARIES "")
set(ARMADILLO_LIBDIRS "")
foreach(l ${ARMADILLO_LIBRARIES_LIST})
  get_filename_component(l_in "${l}" NAME_WE)
  get_filename_component(l_dir "${l}" DIRECTORY)
  string(REGEX REPLACE "^[ ]*lib" "" l_out "${l_in}")
  string(STRIP "${l_dir}" l_dir_out)
  set(ARMADILLO_LIBRARIES "${ARMADILLO_LIBRARIES} ${l_out}")
  set(ARMADILLO_LIBDIRS "${ARMADILLO_LIBDIRS} ${l_dir_out}")
endforeach()

string(STRIP "${ARMADILLO_LIBRARIES}" tmp)
set(ARMADILLO_LIBRARIES "${tmp}")
string(STRIP "${ARMADILLO_LIBDIRS}" tmp)
set(ARMADILLO_LIBDIRS "${tmp}")

configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})
