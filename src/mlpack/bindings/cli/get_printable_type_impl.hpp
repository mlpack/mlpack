/**
 * @file bindings/cli/get_printable_type_impl.hpp
 * @author Ryan Curtin
 *
 * Get the printable type of a parameter.  This type is not the C++ type but
 * instead the command-line type that a user would use.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_TYPE_IMPL_HPP
#define MLPACK_BINDINGS_CLI_GET_PRINTABLE_TYPE_IMPL_HPP

#include "get_printable_type.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

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
    return "flag";
  else if (std::is_same_v<T, int>)
    return "int";
  else if (std::is_same_v<T, double>)
    return "double";
  else if (std::is_same_v<T, std::string>)
    return "string";
  else
    throw std::invalid_argument("unknown parameter type" + data.cppType);
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
    return "int vector";
  else if (std::is_same_v<T, std::vector<std::string>>)
    return "string vector";
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
    return "2-d matrix file";
  else if (std::is_same_v<T, arma::Mat<size_t>>)
    return "2-d index matrix file";
  else if (std::is_same_v<T, arma::rowvec>)
    return "1-d matrix file";
  else if (std::is_same_v<T, arma::Row<size_t>>)
    return "1-d index matrix file";
  else if (std::is_same_v<T, arma::vec>)
    return "1-d matrix file";
  else if (std::is_same_v<T, arma::Col<size_t>>)
    return "1-d index matrix file";
  else
    throw std::invalid_argument("unknown Armadillo type" + data.cppType);
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
  return "2-d categorical matrix file";
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
  return data.cppType + " file";
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
