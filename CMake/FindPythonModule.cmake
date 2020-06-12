# FindPythonModule.cmake: find a specific Python module.
#
# The source here is Mark Moll from a post on the CMake mailing list:
# https://cmake.org/pipermail/cmake/2011-January/041666.html
#
# It has been modified to also check a minimum version if given.
function(find_python_module module)
  string(TOUPPER ${module} module_upper)
  if (NOT PY_${module_upper})
    if (ARGC GREATER 1 AND ARGV1 STREQUAL "REQUIRED")
      set(${module}_FIND_REQUIRED TRUE)
      if (ARGC GREATER 2)
        set(VERSION_REQ ${ARGV2})
      endif ()
    else ()
      if (ARGC GREATER 1)
        # Not required but we have version constraints.
        set(VERSION_REQ ${ARGV1})
      endif ()
    endif ()
    # A module's location is usually a directory, but for binary modules
    # it's a .so file.
    execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" 
      "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
      RESULT_VARIABLE _${module}_status
      OUTPUT_VARIABLE _${module}_location
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if (NOT _${module}_status)
      # Now we have to check the version.
      if (VERSION_REQ)
        execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c"
            "import ${module}; from distutils.version import StrictVersion; print(StrictVersion(${module}.__version__) >= StrictVersion('${VERSION_REQ}'));"
            RESULT_VARIABLE _version_status
            OUTPUT_VARIABLE _version_compare
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        if ("${_version_compare}" STREQUAL "True")
          set(PY_${module_upper} ${_${module}_location} CACHE STRING
            "Location of Python module ${module}")
        else ()
          if (${module_upper}_FIND_REQUIRED)
            message(FATAL_ERROR "Could not find suitable version >= ${VERSION_REQ} of Python module ${module}!")
          else ()
            message(WARNING "Unsuitable version of Python module ${module} (${VERSION_REQ} or greater required).")
          endif ()
        endif ()
      else ()
        # No version requirement so we are done.
        set(PY_${module_upper} ${_${module}_location} CACHE STRING
            "Location of Python module ${module}")
      endif ()
    endif ()
  endif ()
  find_package_handle_standard_args(PY_${module} DEFAULT_MSG PY_${module_upper})
  if (NOT PY_${module_upper}_FOUND AND ${module_upper}_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find Python module ${module}!")
  endif ()
endfunction ()
