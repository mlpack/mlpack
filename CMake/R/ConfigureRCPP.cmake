# ConfigureRCPP.cmake: generate an mlpack .cpp file for a R binding given
# input arguments.
#
# This file depends on the following variables being set:
#
#  * PROGRAM_NAME: name of the binding
#  * PROGRAM_MAIN_FILE: the file containing the mlpackMain() function.
#  * R_CPP_IN: path of the r_method.cpp.in file.
#  * R_CPP_OUT: name of the output .cpp file.
include("${SOURCE_DIR}/CMake/StripType.cmake")
strip_type("${PROGRAM_MAIN_FILE}")

# Extract the required part from *main.cpp.
# Example: mlpack/methods/adaboost/adaboost_main.cpp
string(REGEX REPLACE "${SOURCE_DIR}\\/src\\/" "" INCLUDE_FILE 
    "${PROGRAM_MAIN_FILE}")

file(READ "${MODEL_FILE}" MODEL_FILE_TYPE)
if (NOT (MODEL_FILE_TYPE MATCHES "\"${MODEL_SAFE_TYPES}\""))
  file(APPEND "${MODEL_FILE}" "\"${MODEL_SAFE_TYPES}\"\n")
  # Now, generate the implementation of the functions we need.
  set(MODEL_PTR_IMPLS "")
  list(LENGTH MODEL_TYPES NUM_MODEL_TYPES)
  # Append content to the list.
  if (${NUM_MODEL_TYPES} GREATER 0)
    math(EXPR LOOP_MAX "${NUM_MODEL_TYPES}-1")
    foreach (INDEX RANGE ${LOOP_MAX})
      list(GET MODEL_TYPES ${INDEX} MODEL_TYPE)
      list(GET MODEL_SAFE_TYPES ${INDEX} MODEL_SAFE_TYPE)

      # Define typedef for the model.
      set(MODEL_PTR_TYPEDEF "${MODEL_PTR_TYPEDEF}Rcpp::XPtr<${MODEL_TYPE}>")

      # Generate the implementation.
      set(MODEL_PTR_IMPLS "${MODEL_PTR_IMPLS}
// Get the pointer to a ${MODEL_TYPE} parameter.
// [[Rcpp::export]]
SEXP IO_GetParam${MODEL_SAFE_TYPE}Ptr(const std::string& paramName)
{
  return std::move((${MODEL_PTR_TYPEDEF}) IO::GetParam<${MODEL_TYPE}*>(paramName));
}

// Set the pointer to a ${MODEL_TYPE} parameter.
// [[Rcpp::export]]
void IO_SetParam${MODEL_SAFE_TYPE}Ptr(const std::string& paramName, SEXP ptr)
{
  IO::GetParam<${MODEL_TYPE}*>(paramName) =  Rcpp::as<${MODEL_PTR_TYPEDEF}>(ptr);
  IO::SetPassed(paramName);
}

// Serialize a ${MODEL_TYPE} pointer.
// [[Rcpp::export]]
Rcpp::RawVector Serialize${MODEL_SAFE_TYPE}Ptr(SEXP ptr)
{
  std::ostringstream oss;
  {
    boost::archive::binary_oarchive oa(oss);
    oa << boost::serialization::make_nvp(\"${MODEL_SAFE_TYPE}\",
          *Rcpp::as<${MODEL_PTR_TYPEDEF}>(ptr));
  }

  Rcpp::RawVector raw_vec(oss.str().size());

  // Copy the string buffer so we can return one that won't get deallocated when
  // we exit this function.
  memcpy(&raw_vec[0], oss.str().c_str(), oss.str().size());
  raw_vec.attr(\"type\") = \"${MODEL_SAFE_TYPE}\";
  return raw_vec;
}

// Deserialize a ${MODEL_TYPE} pointer.
// [[Rcpp::export]]
SEXP Deserialize${MODEL_SAFE_TYPE}Ptr(Rcpp::RawVector str)
{
  ${MODEL_TYPE}* ptr = new ${MODEL_TYPE}();

  std::istringstream iss(std::string((char *) &str[0], str.size()));
  {
    boost::archive::binary_iarchive ia(iss);
    ia >> boost::serialization::make_nvp(\"${MODEL_SAFE_TYPE}\", *ptr);
  }

  // R will be responsible for freeing this.
  return std::move((${MODEL_PTR_TYPEDEF})ptr);
}
")
    endforeach ()
  endif()
endif()

# Now configure the files.
configure_file("${R_CPP_IN}" "${R_CPP_OUT}")
