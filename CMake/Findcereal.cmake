#Findcereal.cmake
find_path(CEREAL_INCLUDE_DIRS
  NAMES cereal
  PATHS "$ENV{ProgramFiles}/cereal/include"
  )

if(CEREAL_INCLUDE_DIRS)
  # ------------------------------------------------------------------------
  #  Extract version information from <CEREAL>
  # ------------------------------------------------------------------------

  set(CEREAL_VERSION_MAJOR 0)
  set(CEREAL_VERSION_MINOR 0)
  set(CEREAL_VERSION_PATCH 0)

  if(EXISTS "${CEREAL_INCLUDE_DIRS}/cereal/version.hpp")

    set(CEREAL_FOUND YES)

    # Read and parse cereal version header file for version number
    file(READ "${CEREAL_INCLUDE_DIRS}/cereal/version.hpp"
        _CEREAL_HEADER_CONTENTS)
    string(REGEX REPLACE ".*#define CEREAL_VERSION_MAJOR ([0-9]+).*" "\\1"
        CEREAL_VERSION_MAJOR "${_CEREAL_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define CEREAL_VERSION_MINOR ([0-9]+).*" "\\1"
        CEREAL_VERSION_MINOR "${_CEREAL_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define CEREAL_VERSION_PATCH ([0-9]+).*" "\\1"
        CEREAL_VERSION_PATCH "${_CEREAL_HEADER_CONTENTS}")
  endif()

  set(CEREAL_VERSION_STRING "${CEREAL_VERSION_MAJOR}.${CEREAL_VERSION_MINOR}.${CEREAL_VERSION_PATCH}")
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cereal
  REQUIRED_VARS CEREAL_INCLUDE_DIRS
  VERSION_VAR CEREAL_VERSION_STRING
  )

mark_as_advanced(CEREAL_INCLUDE_DIRS)
