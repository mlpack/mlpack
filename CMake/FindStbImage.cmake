# - Find STB_IMAGE
# Find the STB_IMAGE C++ library
#
# This module sets the following variables:
#  STB_IMAGE_FOUND - set to true if the library is found
#  STB_IMAGE_INCLUDE_DIR - list of required include directories
#  STB_INCLUDE_NEEDS_STB_SUFFIX - whether or not the include files are under an
#     stb/ directory; if "YES", then includes must be done as, e.g.,
#     stb/stb_image.h.

file(GLOB STB_IMAGE_SEARCH_PATHS
    ${CMAKE_BINARY_DIR}/deps/
    ${CMAKE_BINARY_DIR}/deps/stb)
find_path(STB_IMAGE_INCLUDE_DIR_1
    NAMES stb_image.h stb_image_write.h stb_image_resize2.h
    PATHS ${STB_IMAGE_SEARCH_PATHS} ${STB_IMAGE_INCLUDE_DIR})

if(STB_IMAGE_INCLUDE_DIR_1)
  set(STB_IMAGE_INCLUDE_DIR "${STB_IMAGE_INCLUDE_DIR_1}" CACHE PATH
      "stb_image include directory")

  # Either we found /usr/include/stb_image.h (or similar), or the user passed
  # a directory in STB_IMAGE_SEARCH_PATHS that directly contains stb_image.h.
  # In either of those cases, we want to include <stb_image.h>, not
  # <stb/stb_image.h>.
  set(STB_INCLUDE_NEEDS_STB_SUFFIX "NO")
else ()
  find_path(STB_IMAGE_INCLUDE_DIR_2
        NAMES stb_image.h stb_image_write.h stb_image_resize2.h
        PATHS ${STB_IMAGE_SEARCH_PATHS} ${STB_IMAGE_INCLUDE_DIR}
        PATH_SUFFIXES stb/)

  if(STB_IMAGE_INCLUDE_DIR_2)
    set(STB_IMAGE_INCLUDE_DIR "${STB_IMAGE_INCLUDE_DIR_2}" CACHE PATH
        "stb_image include directory")

    # Since we searched the same paths but allowed an stb/ suffix this time,
    # then there is definitely a suffix.
    set(STB_INCLUDE_NEEDS_STB_SUFFIX "YES")
    # Strip the suffix.
    string(REGEX REPLACE ".*stb[/]?$" "" STB_IMAGE_INCLUDE_DIR
        "${STB_IMAGE_INCLUDE_DIR}")
  endif ()
endif ()

# Checks 'REQUIRED'.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(StbImage REQUIRED_VARS STB_IMAGE_INCLUDE_DIR)

mark_as_advanced(STB_IMAGE_INCLUDE_DIR)
