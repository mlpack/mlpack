# ConfigureJuliaHCPP.cmake: generate an mlpack .h file for a Julia binding given
# input arguments.
#
# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * JULIA_H_IN: path of the julia_method.h.in file.
#  * JULIA_H_OUT: name of the output .h file.
#  * JULIA_CPP_IN: path of the julia_method.cpp.in file.
#  * JULIA_CPP_OUT: name of the output .cpp file.
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
// Get the pointer to a ${MODEL_TYPE} parameter.
void* IO_GetParam${MODEL_SAFE_TYPE}Ptr(const char* paramName);
// Set the pointer to a ${MODEL_TYPE} parameter.
void IO_SetParam${MODEL_SAFE_TYPE}Ptr(const char* paramName, void* ptr);
// Serialize a ${MODEL_TYPE} pointer.
char* Serialize${MODEL_SAFE_TYPE}Ptr(void* ptr, size_t* length);
// Deserialize a ${MODEL_TYPE} pointer.
void* Deserialize${MODEL_SAFE_TYPE}Ptr(const char* buffer, const size_t length);
")

    # Generate the implementation.
    set(MODEL_PTR_IMPLS "${MODEL_PTR_IMPLS}
// Get the pointer to a ${MODEL_TYPE} parameter.
void* IO_GetParam${MODEL_SAFE_TYPE}Ptr(const char* paramName)
{
  return (void*) IO::GetParam<${MODEL_TYPE}*>(paramName);
}

// Set the pointer to a ${MODEL_TYPE} parameter.
void IO_SetParam${MODEL_SAFE_TYPE}Ptr(const char* paramName, void* ptr)
{
  IO::GetParam<${MODEL_TYPE}*>(paramName) = (${MODEL_TYPE}*) ptr;
  IO::SetPassed(paramName);
}

// Serialize a ${MODEL_TYPE} pointer.
char* Serialize${MODEL_SAFE_TYPE}Ptr(void* ptr, size_t* length)
{
  std::ostringstream oss;
  {
    boost::archive::binary_oarchive oa(oss);
    ${MODEL_TYPE}* model = (${MODEL_TYPE}*) ptr;
    oa << boost::serialization::make_nvp(\"${MODEL_SAFE_TYPE}\", model);
  }

  *length = oss.str().length();

  // Copy the string buffer so we can return one that won't get deallocated when
  // we exit this function.  Julia will be responsible for freeing this.
  char* buffer = new char[*length];
  memcpy(buffer, oss.str().data(), *length);
  return buffer;
}

// Deserialize a ${MODEL_TYPE} pointer.
void* Deserialize${MODEL_SAFE_TYPE}Ptr(const char* buffer, const size_t length)
{
  ${MODEL_TYPE}* t = new ${MODEL_TYPE}();

  std::istringstream iss(std::string(buffer, length));
  {
    boost::archive::binary_iarchive ia(iss);
    ia >> boost::serialization::make_nvp(\"${MODEL_SAFE_TYPE}\", t);
  }

  // Julia will be responsible for freeing this.
  return (void*) t;
}
")
  endforeach ()
endif()

# Now configure both of the files.
configure_file("${JULIA_H_IN}" "${JULIA_H_OUT}")
configure_file("${JULIA_CPP_IN}" "${JULIA_CPP_OUT}")
