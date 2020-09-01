# AppendModel.cmake: append model definition and gettter setter methods for
# mlpack model types to the existing file of models.go.

# This function depends on the following variables being set:
#
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * SERIALIZATION_FILE: file to append types to
#
function(append_model SERIALIZATION_FILE PROGRAM_MAIN_FILE)
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

      # Convert the model type similar to goStrippedType(bindings/go/strip_type.hpp).
      string(LENGTH ${MODEL_SAFE_TYPE} NUM_MODEL_CHAR)
      if (${NUM_MODEL_CHAR} GREATER 0)
        math(EXPR LAST_CHAR_INDEX "${NUM_MODEL_CHAR}-1")
        set(BREAK 0)
        foreach (INDEX RANGE ${LAST_CHAR_INDEX})
          if (NOT "${MODEL_SAFE_TYPE}" MATCHES "[^A-Z]")
            string(TOLOWER ${MODEL_SAFE_TYPE} GOMODEL_SAFE_TYPE)
            break()
          endif()
          string(SUBSTRING ${MODEL_SAFE_TYPE} "${INDEX}" "1" MODEL_CHAR)
          if (${BREAK} EQUAL 0)
            string(TOLOWER ${MODEL_CHAR} MODEL_CHAR)
            string(APPEND GOMODEL_SAFE_TYPE ${MODEL_CHAR})
            math(EXPR INDEX1 "${INDEX}+1")
            math(EXPR INDEX2 "${INDEX}+2")
            string(SUBSTRING "${MODEL_SAFE_TYPE}" "${INDEX1}" "1" MODEL_CHAR1)
            string(SUBSTRING "${MODEL_SAFE_TYPE}" "${INDEX2}" "1" MODEL_CHAR2)
            if ("${MODEL_CHAR1}" MATCHES "[A-Z]" AND "${MODEL_CHAR2}" MATCHES "[^A-Z]")
              set(BREAK 1)
            endif()
          else ()
            string(APPEND GOMODEL_SAFE_TYPE ${MODEL_CHAR})
          endif()
        endif()
     endforeach()

      # See if the model type already exists.
      file(READ "${SERIALIZATION_FILE}" SERIALIZATION_FILE_CONTENTS)
      string(FIND
          "${SERIALIZATION_FILE_CONTENTS}"
          "type ${GOMODEL_SAFE_TYPE} struct {\n"
          FIND_OUT)

      # If it doesn't exist, append it.
      if (${FIND_OUT} EQUAL -1)
        # Now append the type to the list of types, and define any serialization
        # function.
        file(APPEND
            "${SERIALIZATION_FILE}"
            "type ${GOMODEL_SAFE_TYPE} struct {\n"
            "  mem unsafe.Pointer \n"
            "}\n\n"
            "func (m *${GOMODEL_SAFE_TYPE}) alloc"
            "${MODEL_SAFE_TYPE}(identifier string) {\n"
            "  m.mem = C.mlpackGet${MODEL_SAFE_TYPE}Ptr(C.CString(identifier))\n"
            "  runtime.KeepAlive(m)\n"
            "}\n\n"
            "func (m *${GOMODEL_SAFE_TYPE}) get"
            "${MODEL_SAFE_TYPE}(identifier string) {\n"
            "  m.alloc${MODEL_SAFE_TYPE}(identifier)\n"
            "}\n\n"
            "func set${MODEL_SAFE_TYPE}(identifier string, ptr *"
            "${GOMODEL_SAFE_TYPE}) {\n"
            " C.mlpackSet${MODEL_SAFE_TYPE}"
            "Ptr(C.CString(identifier), (unsafe.Pointer)(ptr.mem))\n"
            "}\n\n")
      endif ()
    endforeach ()
  endif()
endfunction()
