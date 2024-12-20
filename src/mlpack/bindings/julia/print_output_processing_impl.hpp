/**
 * @file bindings/julia/print_output_processing_impl.hpp
 * @author Ryan Curtin
 *
 * Print Julia code to handle output arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_OUTPUT_PROCESSING_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_OUTPUT_PROCESSING_IMPL_HPP

#include "print_output_processing.hpp"

#include <mlpack/bindings/util/strip_type.hpp>
#include "get_julia_type.hpp"

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Print the output processing (basically calling params.GetParam<>()) for a
 * non-serializable type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const std::string& /* functionName */,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<!data::HasSerialize<T>::value>*,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  std::string type;
  if (std::is_same_v<T, bool>)
    type = "Bool";
  else if (std::is_same_v<T, int>)
    type = "Int";
  else if (std::is_same_v<T, double>)
    type = "Double";
  else if (std::is_same_v<T, std::string>)
    type = "String";
  else if (std::is_same_v<T, std::vector<std::string>>)
    type = "VectorStr";
  else if (std::is_same_v<T, std::vector<int>>)
    type = "VectorInt";
  else
    type = "Unknown";

  // Strings need a little special handling.
  if (std::is_same_v<T, std::string>)
    std::cout << "Base.unsafe_string(";

  std::cout << "GetParam" << type << "(p, \"" << d.name << "\")";

  if (std::is_same_v<T, std::string>)
    std::cout << ")";
}

/**
 * Print the output processing for an Armadillo type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const std::string& /* functionName */,
    const std::enable_if_t<arma::is_arma_type<T>::value>*,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  std::string uChar = (std::is_same_v<typename T::elem_type, size_t>) ?
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

  std::cout << "GetParam" << uChar << matTypeSuffix << "(p, \"" << d.name
      << "\"" << extra << ", juliaOwnedMemory)";
}

/**
 * Print the output processing for a serializable type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const std::string& functionName,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<data::HasSerialize<T>::value>*,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  std::string type = util::StripType(d.cppType);
  std::cout << functionName << "_internal.GetParam"
      << type << "(p, \"" << d.name << "\", modelPtrs)";
}

/**
 * Print the output processing for a mat/DatasetInfo tuple type.
 */
template<typename T>
void PrintOutputProcessing(
    util::ParamData& d,
    const std::string& /* functionName */,
    const std::enable_if_t<std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  std::cout << "GetParamMatWithInfo(p, \"" << d.name << "\", juliaOwnedMemory)";
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
