# - Find STB_IMAGE
# Find the STB_IMAGE C++ library
#
# This module sets the following variables:
#  STB_IMAGE_FOUND - set to true if the library is found
#  STB_IMAGE_INCLUDE_DIR - list of required include directories

file(GLOB STB_IMAGE_SEARCH_PATHS
    ${CMAKE_BINARY_DIR}/deps/
    ${CMAKE_BINARY_DIR}/deps/stb)
find_path(STB_IMAGE_INCLUDE_DIR
    NAMES stb/stb_image.h stb/stb_image_write.h
    PATHS ${STB_IMAGE_SEARCH_PATHS})

if(STB_IMAGE_INCLUDE_DIR)
  set(STB_IMAGE_FOUND YES)
else ()
  find_path(STB_IMAGE_INCLUDE_DIR
        NAMES stb_image.h stb_image_write.h
        PATHS ${STB_IMAGE_SEARCH_PATHS})

  if(STB_IMAGE_INCLUDE_DIR)
    set(STB_IMAGE_FOUND YES)
  endif ()
endif ()

# Make sure that stb/ is the last part of the include directory, if it was
# found.
if (STB_IMAGE_INCLUDE_DIR)
  string(REGEX MATCH ".*stb[/]?" STB_INCLUDE_HAS_TRAILING_STB
      "${STB_IMAGE_INCLUDE_DIR}")
  if (NOT STB_INCLUDE_HAS_TRAILING_STB)
    set(STB_IMAGE_INCLUDE_DIR "${STB_IMAGE_INCLUDE_DIR}/stb/")
  endif ()
endif ()

# Checks 'REQUIRED'.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(StbImage
    REQUIRED_VARS STB_IMAGE_INCLUDE_DIR)

mark_as_advanced(STB_IMAGE_INCLUDE_DIR)
