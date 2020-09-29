# ConfigureGoHCPP.cmake: generate an mlpack .h/.cpp file for a Go binding given
# input arguments.
#
# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * GO_IN: path of the go_method.h.in/go_method.cpp.in file.
#  * GO_OUT: name of the output .h/.cpp file.
#
include("${SOURCE_DIR}/CMake/StripType.cmake")
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

    # Generate the definition.
    set(MODEL_PTR_DEFNS "${MODEL_PTR_DEFNS}
// Set the pointer to a ${MODEL_TYPE} parameter.
extern void mlpackSet${MODEL_SAFE_TYPE}Ptr(const char* identifier, void* value);

// Get the pointer to a ${MODEL_TYPE} parameter.
extern void* mlpackGet${MODEL_SAFE_TYPE}Ptr(const char* identifier);
"
)

    # Generate the implementation.
    set(MODEL_PTR_IMPLS "${MODEL_PTR_IMPLS}
// Set the pointer to a ${MODEL_TYPE} parameter.
extern \"C\"  void mlpackSet${MODEL_SAFE_TYPE}Ptr(
    const char* identifier,
    void* value)
{
  mlpack::util::SetParamPtr<${MODEL_TYPE}>(identifier,
  static_cast<${MODEL_TYPE}*>(value));
}

// Get the pointer to a ${MODEL_TYPE} parameter.
extern \"C\" void *mlpackGet${MODEL_SAFE_TYPE}Ptr(const char* identifier)
{
  ${MODEL_TYPE} *modelptr = IO::GetParam<${MODEL_TYPE}*>(identifier);
  return modelptr;
}
")
  endforeach ()
endif()

# Convert ${PROGRAM_NAME} from snake_case to CamelCase.
string(LENGTH ${PROGRAM_NAME} NUM_MODEL_CHAR)
if (${NUM_MODEL_CHAR} GREATER 0)
  math(EXPR LAST_CHAR_INDEX "${NUM_MODEL_CHAR}-2")
  string(SUBSTRING ${PROGRAM_NAME} "0" "1" MODEL_CHAR)
  string(TOUPPER ${MODEL_CHAR} MODEL_CHAR)
  string(APPEND GOPROGRAM_NAME ${MODEL_CHAR})
  foreach (INDEX0 RANGE ${LAST_CHAR_INDEX})
    math(EXPR INDEX0 "${INDEX0}+1") 
    math(EXPR INDEX1 "${INDEX0}+1")
    math(EXPR INDEX2 "${INDEX0}-1")
    string(SUBSTRING "${PROGRAM_NAME}" "${INDEX0}" "1" MODEL_CHAR1)
    string(SUBSTRING "${PROGRAM_NAME}" "${INDEX1}" "1" MODEL_CHAR2)
    string(SUBSTRING "${PROGRAM_NAME}" "${INDEX2}" "1" MODEL_CHAR3)
    if ("${MODEL_CHAR1}" MATCHES "_")
       string(TOUPPER ${MODEL_CHAR2} MODEL_CHAR2)
       string(APPEND GOPROGRAM_NAME ${MODEL_CHAR2})
       set(INDEX0 ${INDEX1})
    elseif ("${MODEL_CHAR3}" MATCHES "_")
       continue()
    else()
       string(APPEND GOPROGRAM_NAME ${MODEL_CHAR1})
    endif()
   endforeach()
endif()

# Now configure the files.
configure_file("${GO_IN}" "${GO_OUT}")
