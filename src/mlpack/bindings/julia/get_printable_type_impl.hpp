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
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::disable_if<util::IsStdVector<T>>::type*,
    const typename boost::disable_if<data::HasSerialize<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  if (std::is_same<T, bool>::value)
    return "Bool";
  else if (std::is_same<T, int>::value)
    return "Int";
  else if (std::is_same<T, double>::value)
    return "Float64";
  else if (std::is_same<T, std::string>::value)
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
    const typename std::enable_if<util::IsStdVector<T>::value>::type*)
{
  if (std::is_same<T, std::vector<int>>::value)
    return "Array{Int, 1}";
  else if (std::is_same<T, std::vector<std::string>>::value)
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
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*)
{
  if (std::is_same<T, arma::mat>::value)
    return "Float64 matrix-like";
  else if (std::is_same<T, arma::Mat<size_t>>::value)
    return "Int matrix-like";
  else if (std::is_same<T, arma::rowvec>::value)
    return "Float64 vector-like";
  else if (std::is_same<T, arma::Row<size_t>>::value)
    return "Int vector-like";
  else if (std::is_same<T, arma::vec>::value)
    return "Float64 vector-like";
  else if (std::is_same<T, arma::Col<size_t>>::value)
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
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "Tuple{Array{Bool, 1}, Array{Float64, 2}}";
}

/**
 * Return a string representing the command-line type of a model.
 */
template<typename T>
std::string GetPrintableType(
    util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::enable_if<data::HasSerialize<T>>::type*)
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
