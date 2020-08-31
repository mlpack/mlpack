# AppendSerialization.cmake: append imports for serialization and
# deserialization for mlpack model types to the existing list of serialization
# and deserialization imports.

# This function depends on the following variables being set:
#
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * SERIALIZATION_FILE: file to append types to
#  * SERIALIZE: It is of bool type. If SERIALIZE is true we have to print
#               Serialize, else Deserialize.
#
function(append_serialization SERIALIZATION_FILE PROGRAM_MAIN_FILE SERIALIZE)
  include("${CMAKE_SOURCE_DIR}/CMake/StripType.cmake")
  strip_type("${PROGRAM_MAIN_FILE}")

  list(LENGTH MODEL_TYPES NUM_MODEL_TYPES)
  if (${NUM_MODEL_TYPES} GREATER 0)
    math(EXPR LOOP_MAX "${NUM_MODEL_TYPES}-1")
    foreach (INDEX RANGE ${LOOP_MAX})
      list(GET MODEL_TYPES ${INDEX} MODEL_TYPE)
      list(GET MODEL_SAFE_TYPES ${INDEX} MODEL_SAFE_TYPE)
      file(READ "${SERIALIZATION_FILE}" SERIALIZATION_FILE_CONTENTS)
      if (SERIALIZE)
        # See if the model type already exists.
        string(FIND
            "${SERIALIZATION_FILE_CONTENTS}"
            "\"${MODEL_SAFE_TYPE}\" = Serialize${MODEL_SAFE_TYPE}Ptr,"
            FIND_OUT)

        # If it doesn't exist, append it.
        if (${FIND_OUT} EQUAL -1)
          # Now append the type to the list of types, and define any serialization
          # function.
          file(APPEND
              "${SERIALIZATION_FILE}"
              "      \"${MODEL_SAFE_TYPE}\" = Serialize${MODEL_SAFE_TYPE}Ptr,\n")
        endif()
      elseif (NOT SERIALIZE)
        # See if the model type already exists.
        string(FIND
            "${SERIALIZATION_FILE_CONTENTS}"
            "\"${MODEL_SAFE_TYPE}\" = Deserialize${MODEL_SAFE_TYPE}Ptr,"
            FIND_OUT)

        # If it doesn't exist, append it.
        if (${FIND_OUT} EQUAL -1)
          # Now append the type to the list of types, and define any deserialization
          # function.
          file(APPEND
              "${SERIALIZATION_FILE}"
              "      \"${MODEL_SAFE_TYPE}\" = Deserialize${MODEL_SAFE_TYPE}Ptr,\n")
        endif()
      endif()
    endforeach ()
  endif()
endfunction()
