/**
 * @file bindings/julia/get_printable_type_impl.hpp
 * @author Ryan Curtin
 *
 * Get the printable type of a parameter.  This type is not the C++ type but
 * instead the Julia type that a user would use.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_GET_PRINTABLE_TYPE_IMPL_HPP
#define MLPACK_BINDINGS_JULIA_GET_PRINTABLE_TYPE_IMPL_HPP

#include "get_printable_type.hpp"
#include <mlpack/bindings/util/strip_type.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Return a string representing the command-line type of an option.
 */
template<typename T>
std::string GetPrintableType(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<!util::IsStdVector<T>::value>*,
    const std::enable_if_t<!data::HasSerialize<T>::value>*,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  if (std::is_same_v<T, bool>)
    return "Bool";
  else if (std::is_same_v<T, int>)
    return "Int";
  else if (std::is_same_v<T, double>)
    return "Float64";
  else if (std::is_same_v<T, std::string>)
    return "String";
  else
    throw std::invalid_argument("unknown parameter type " + data.cppType);
}

/**
 * Return a string representing the command-line type of a vector.
 */
template<typename T>
std::string GetPrintableType(
    util::ParamData& data,
    const std::enable_if_t<util::IsStdVector<T>::value>*)
{
  if (std::is_same_v<T, std::vector<int>>)
    return "Array{Int, 1}";
  else if (std::is_same_v<T, std::vector<std::string>>)
    return "Array{String, 1}";
  else
    throw std::invalid_argument("unknown vector type " + data.cppType);
}

/**
 * Return a string representing the command-line type of a matrix option.
 */
template<typename T>
std::string GetPrintableType(
    util::ParamData& data,
    const std::enable_if_t<arma::is_arma_type<T>::value>*)
{
  if (std::is_same_v<T, arma::mat>)
    return "Float64 matrix-like";
  else if (std::is_same_v<T, arma::Mat<size_t>>)
    return "Int matrix-like";
  else if (std::is_same_v<T, arma::rowvec>)
    return "Float64 vector-like";
  else if (std::is_same_v<T, arma::Row<size_t>>)
    return "Int vector-like";
  else if (std::is_same_v<T, arma::vec>)
    return "Float64 vector-like";
  else if (std::is_same_v<T, arma::Col<size_t>>)
    return "Int vector-like";
  else
    throw std::invalid_argument("unknown Armadillo type " + data.cppType);
}

/**
 * Return a string representing the command-line type of a matrix tuple option.
 */
template<typename T>
std::string GetPrintableType(
    util::ParamData& /* data */,
    const std::enable_if_t<std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "Tuple{Array{Bool, 1}, Array{Float64, 2}}";
}

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string GetPrintableType(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<data::HasSerialize<T>::value>*)
{
  std::string type = util::StripType(data.cppType);
  if (type == "mlpackModel")
  {
    // If this is true, then we are being called from the Markdown bindings.
    // This will be printed as the general documentation for model types.
    return "<Model> (mlpack model)";
  }
  else
  {
    return type;
  }
}

} // namespace julia
} // namespace bindings
} // namespace mlpack

#endif
