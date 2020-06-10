# Set of helper functions for our CMake

# Joins arguments and places the results in ${result_var}.
function(join result_var)
  set(result)
  foreach (arg ${ARGN})
    set(result "${result}${arg}")
  endforeach ()
  set(${result_var} "${result}" PARENT_SCOPE)
endfunction()

# Sets a cache variable with a docstring joined from multiple arguments:
#   set(<variable> <value>... CACHE <type> <docstring>...)
# This allows splitting a long docstring for readability.
function(set_verbose)
  cmake_parse_arguments(SET_VERBOSE "" "" "CACHE" ${ARGN})
  list(GET SET_VERBOSE_CACHE 0 type)
  list(REMOVE_AT SET_VERBOSE_CACHE 0)
  join(doc ${SET_VERBOSE_CACHE})
  set(${SET_VERBOSE_UNPARSED_ARGUMENTS} CACHE ${type} ${doc})
endfunction()
