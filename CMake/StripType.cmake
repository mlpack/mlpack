# StripType.cmake: Extract ModelType from the main file and turn it into
# something that has no special characters that can simply be used.

# This function depends on the following variables being set:
#
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#
function(strip_type PROGRAM_MAIN_FILE)
  # We need to parse the main file and find any PARAM_MODEL_* lines.
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

  set(MODEL_TYPES ${MODEL_TYPES} PARENT_SCOPE)
  set(MODEL_SAFE_TYPES ${MODEL_SAFE_TYPES} PARENT_SCOPE)
endfunction()
