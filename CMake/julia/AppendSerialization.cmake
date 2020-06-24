# AppendSerialization.cmake: append imports for serialization and
# deserialization for mlpack model types to the existing list of serialization
# and deserialization imports.

# This function depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * SERIALIZATION_FILE: file to append types to
#
function(append_serialization SERIALIZATION_FILE PROGRAM_NAME PROGRAM_MAIN_FILE)
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
      file(READ "${SERIALIZATION_FILE}" SERIALIZATION_FILE_CONTENTS)
      string(FIND
          "${SERIALIZATION_FILE_CONTENTS}"
          "serialize_bin(stream::IO, model::${MODEL_SAFE_TYPE})"
          FIND_OUT)

      # If it doesn't exist, append it.
      if (${FIND_OUT} EQUAL -1)
        # Now append the type to the list of types, and define any serialization
        # function.
        file(APPEND
            "${SERIALIZATION_FILE}"
            "serialize_bin(stream::IO, model::${MODEL_SAFE_TYPE}) =\n"
            "    _Internal.${PROGRAM_NAME}_internal.serialize${MODEL_SAFE_TYPE}(stream, model)\n"
            "deserialize_bin(stream::IO, ::Type{${MODEL_SAFE_TYPE}}) =\n"
            "    _Internal.${PROGRAM_NAME}_internal.deserialize${MODEL_SAFE_TYPE}(stream)\n"
            "\n"
            "function Serialization.serialize(s::Serialization.AbstractSerializer,\n"
            "                                 model::${MODEL_SAFE_TYPE})\n"
            "  Serialization.writetag(s.io, Serialization.OBJECT_TAG)\n"
            "  Serialization.serialize(s, ${MODEL_SAFE_TYPE})\n"
            "  serialize_bin(s.io, model)\n"
            "end\n"
            "\n"
            "function Serialization.deserialize(s::Serialization.AbstractSerializer,\n"
            "                                   ::Type{${MODEL_SAFE_TYPE}})\n"
            "  deserialize_bin(s.io, ${MODEL_SAFE_TYPE})\n"
            "end\n"
            "\n")
      endif ()
    endforeach ()
  endif()
endfunction()
