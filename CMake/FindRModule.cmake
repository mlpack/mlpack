# FindRModule.cmake: find a specific R module.
function(find_r_module module)
  string(TOUPPER ${module} module_upper)
  if (NOT R_${module_upper})
    if (ARGC GREATER 1)
      # Not required but we have version constraints.
      set(VERSION_REQ ${ARGV1})
    endif ()
    # A module's location is usually a directory, but for binary modules
    # it's a .so file.
    execute_process(COMMAND ${RSCRIPT_EXECUTABLE} "-e" "find.package('${module}')"
      RESULT_VARIABLE _${module}_status
      OUTPUT_VARIABLE _${module}_location
      ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

   # Some cleanup in location of R Module.
   string(REGEX MATCHALL "\".*\""  _${module}_location "${_${module}_location}" )
   string(REGEX REPLACE "\"" "" _${module}_location "${_${module}_location}" )

    if (NOT _${module}_status)
      # Now we have to check the version.
      if (VERSION_REQ)
        execute_process(COMMAND ${RSCRIPT_EXECUTABLE} "-e" "packageVersion('${module}')"
            RESULT_VARIABLE _version_status
            OUTPUT_VARIABLE _version_compare
            OUTPUT_STRIP_TRAILING_WHITESPACE)

        string(REGEX MATCHALL "‘[0-9._]*’" _version_compare "${_version_compare}")
        string(REGEX REPLACE "‘" "" _version_compare "${_version_compare}")
        string(REGEX REPLACE "’" "" _version_compare "${_version_compare}")
        if ("${_version_compare}" GREATER_EQUAL "${VERSION_REQ}")
          set(R_${module_upper}
            "${_${module}_location} (found suitable version \"${_version_compare}\", minimum required is \"${VERSION_REQ}\")"
            CACHE STRING "Location of R module ${module}"
          )
        else ()
          message(WARNING "Unsuitable version of R module ${module} (${VERSION_REQ} or greater required).")
        endif ()
      else ()
        # No version requirement so we are done.
        set(R_${module_upper} ${_${module}_location} CACHE STRING "Location of R module ${module}")
      endif ()
    endif ()
  endif ()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(R_${module} DEFAULT_MSG R_${module_upper})
endfunction ()
