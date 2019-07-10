/**
 * @file print_output_processing_impl.hpp
 * @author Ryan Curtin
 *
 * Print Julia code to handle output arguments.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_OUTPUT_PROCESSING_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_OUTPUT_PROCESSING_IMPL_HPP

#include "print_output_processing.hpp"

#include "strip_type.hpp"
#include "get_julia_type.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the output processing (basically calling CLI::GetParam<>()) for a
 * non-serializable type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const std::string& /* functionName */,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  std::string type;
  if (std::is_same<T, bool>::value)
    type = "Bool";
  else if (std::is_same<T, int>::value)
    type = "Int";
  else if (std::is_same<T, double>::value)
    type = "Double";
  else if (std::is_same<T, std::string>::value)
    type = "String";
  else if (std::is_same<T, std::vector<std::string>>::value)
    type = "VectorStr";
  else if (std::is_same<T, std::vector<int>>::value)
    type = "VectorInt";
  else
    type = "Unknown";

  // Strings need a little special handling.
  if (std::is_same<T, std::string>::value)
    std::cout << "Base.unsafe_string(";

  std::cout << "CLIGetParam" << type << "(\"" << d.name << "\")";

  if (std::is_same<T, std::string>::value)
    std::cout << ")";
}

/**
 * Print the output processing for an Armadillo type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const std::string& /* functionName */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  std::string uChar = (std::is_same<typename T::elem_type, size_t>::value) ?
      "U" : "";
  std::string matTypeSuffix = "";
  std::string extra = "";
  if (T::is_row)
  {
    matTypeSuffix = "Row";
  }
  else if (T::is_col)
  {
    matTypeSuffix = "Col";
  }
  else
  {
    matTypeSuffix = "Mat";
    extra = ", points_are_rows";
  }

  std::cout << "CLIGetParam" << uChar << matTypeSuffix << "(\"" << d.name
      << "\"" << extra << ")";
}

/**
 * Print the output processing for a serializable type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const std::string& functionName,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  std::cout << functionName << "_internal.CLIGetParam" << StripType(d.cppType)
      << "Ptr(\"" << d.name << "\")";
}

/**
 * Print the output processing for a mat/DatasetInfo tuple type.
 */
template<typename T>
void PrintOutputProcessing(
    const util::ParamData& d,
    const std::string& /* functionName */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  std::cout << "CLIGetParamMatWithInfo(\"" << d.name << "\")";
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
