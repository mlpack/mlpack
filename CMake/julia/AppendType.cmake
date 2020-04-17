# AppendType.cmake: append a Julia mlpack model type to the list of Julia mlpack
# model types.

# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * TYPES_FILE: file to append types to
#
# We need to parse the main file and find any PARAM_MODEL_* lines.
function(append_type TYPES_FILE PROGRAM_NAME PROGRAM_MAIN_FILE)
  file(READ "${PROGRAM_MAIN_FILE}" MAIN_FILE)

  # Grab all "PARAM_MODEL_IN(Model,", "PARAM_MODEL_IN_REQ(Model,",
  # "PARAM_MODEL_OUT(Model,".
  string(REGEX MATCHALL "PARAM_MODEL_IN\\([A-Za-z_<>]*," MODELS_IN
      "${MAIN_FILE}")
  string(REGEX MATCHALL "PARAM_MODEL_IN_REQ\\([A-Za-z_<>]*," MODELS_IN_REQ
      "${MAIN_FILE}")
  string(REGEX MATCHALL "PARAM_MODEL_OUT\\([A-Za-z_]*," MODELS_OUT "${MAIN_FILE}")

  string(REGEX REPLACE "PARAM_MODEL_IN\\(" "" MODELS_IN_STRIP1 "${MODELS_IN}")
  string(REGEX REPLACE "," "" MODELS_IN_STRIP2 "${MODELS_IN_STRIP1}")
  string(REGEX REPLACE "[<>,]" "" MODELS_IN_SAFE_STRIP2 "${MODELS_IN_STRIP1}")

  string(REGEX REPLACE "PARAM_MODEL_IN_REQ\\(" "" MODELS_IN_REQ_STRIP1
      "${MODELS_IN_REQ}")
  string(REGEX REPLACE "," "" MODELS_IN_REQ_STRIP2 "${MODELS_IN_REQ_STRIP1}")
  string(REGEX REPLACE "[<>,]" "" MODELS_IN_REQ_SAFE_STRIP2
      "${MODELS_IN_REQ_STRIP1}")

  string(REGEX REPLACE "PARAM_MODEL_OUT\\(" "" MODELS_OUT_STRIP1 "${MODELS_OUT}")
  string(REGEX REPLACE "," "" MODELS_OUT_STRIP2 "${MODELS_OUT_STRIP1}")
  string(REGEX REPLACE "[<>,]" "" MODELS_OUT_SAFE_STRIP2 "${MODELS_OUT_STRIP1}")

  set(MODEL_TYPES ${MODELS_IN_STRIP2} ${MODELS_IN_REQ_STRIP2}
      ${MODELS_OUT_STRIP2})
  set(MODEL_SAFE_TYPES ${MODELS_IN_SAFE_STRIP2} ${MODELS_IN_REQ_SAFE_STRIP2}
      ${MODELS_OUT_SAFE_STRIP2})
  if (MODEL_TYPES)
    list(REMOVE_DUPLICATES MODEL_TYPES)
  endif ()
  if (MODEL_SAFE_TYPES)
    list(REMOVE_DUPLICATES MODEL_SAFE_TYPES)
  endif ()

  # Now, generate the definitions of the functions we need.
  set(MODEL_PTR_DEFNS "")
  set(MODEL_PTR_IMPLS "")
  list(LENGTH MODEL_TYPES NUM_MODEL_TYPES)
  if (${NUM_MODEL_TYPES} GREATER 0)
    message(STATUS "num_model_types is ${NUM_MODEL_TYPES}!")
    foreach (INDEX RANGE ${NUM_MODEL_TYPES})
      list(GET MODEL_TYPES ${INDEX} MODEL_TYPE)
      list(GET MODEL_SAFE_TYPES ${INDEX} MODEL_SAFE_TYPE)

      # See if the model type already exists.
      file(READ "${TYPES_FILE}" TYPES_FILE_CONTENTS)
      string(FIND "${TYPES_FILE_CONTENTS}" "struct ${MODEL_SAFE_TYPE}Ptr"
          FIND_OUT)

      # If it doesn't exist, append it.
      if (${FIND_OUT} EQUAL -1)
        # Now append the type to the list of types, and define any serialization
        # function.
        file(APPEND
            "${TYPES_FILE}"
            "struct ${MODEL_SAFE_TYPE}Ptr\n"
            "  ptr::Ptr{Nothing}\n"
            "end\n"
            "\n")
      endif ()
    endforeach ()
  endif()
endfunction()
