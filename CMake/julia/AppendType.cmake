# AppendType.cmake: append a Julia mlpack model type to the list of Julia mlpack
# model types.

# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * TYPES_FILE: file to append types to
#
function(append_type TYPES_FILE PROGRAM_NAME PROGRAM_MAIN_FILE)
  include("${CMAKE_SOURCE_DIR}/CMake/StripType.cmake")
  strip_type("${PROGRAM_MAIN_FILE}")

  # Now, generate the definitions of the functions we need.
  set(MODEL_PTR_DEFNS "")
  set(MODEL_PTR_IMPLS "")
  list(LENGTH MODEL_TYPES NUM_MODEL_TYPES)
  if (${NUM_MODEL_TYPES} GREATER 0)
    math(EXPR LOOP_MAX "${NUM_MODEL_TYPES}-1")
    foreach (INDEX RANGE ${LOOP_MAX})
      list(GET MODEL_TYPES ${INDEX} MODEL_TYPE)
      list(GET MODEL_SAFE_TYPES ${INDEX} MODEL_SAFE_TYPE)

      # See if the model type already exists.
      file(READ "${TYPES_FILE}" TYPES_FILE_CONTENTS)
      string(FIND "${TYPES_FILE_CONTENTS}" "struct ${MODEL_SAFE_TYPE}"
          FIND_OUT)

      # If it doesn't exist, append it.
      if (${FIND_OUT} EQUAL -1)
        # Now append the type to the list of types, and define any serialization
        # function.
        file(APPEND
            "${TYPES_FILE}"
            "struct ${MODEL_SAFE_TYPE}\n"
            "  ptr::Ptr{Nothing}\n"
            "end\n"
            "\n")
      endif ()
    endforeach ()
  endif()
endfunction()
