# This file is orignially written by: 
# https://github.com/facebookresearch/flashlight

# Try to find Cereal
#
# Sets the following variables:
# CEREAL_FOUND
# CEREAL_INCLUDE_DIRS - directories with Cereal headers
# CEREAL_DEFINITIONS - Cereal compiler flags

find_path(cereal_header_paths_tmp
  NAMES
    cereal.hpp
  PATH_SUFFIXES
  include
  cereal/include
	PATHS
    ${CEREAL_ROOT_DIR}
    ${CEREAL_ROOT_DIR}/include
    ${CEREAL_ROOT_DIR}/cereal/include
    $ENV{CEREAL_ROOT_DIR}
    $ENV{CEREAL_ROOT_DIR}/include
    $ENV{CEREAL_ROOT_DIR}/cereal
    )

get_filename_component(cereal_INCLUDE_DIRS ${cereal_header_paths_tmp} PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cereal
  REQUIRED_VARS cereal_INCLUDE_DIRS
  )

mark_as_advanced(cereal_FOUND)
