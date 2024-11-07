/**
 * @file bindings/cli/get_printable_param_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of parameter printing functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_IMPL_HPP
#define MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_IMPL_HPP

#include "get_printable_param.hpp"
#include "get_param.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

//! Print an option.
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<!util::IsStdVector<T>::value>*,
    const std::enable_if_t<!data::HasSerialize<T>::value>*,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  std::ostringstream oss;
  oss << std::any_cast<T>(data.value);
  return oss.str();
}

//! Print a vector option.
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<util::IsStdVector<T>::value>* /* junk */)
{
  const T& t = std::any_cast<T>(data.value);

  std::ostringstream oss;
  for (size_t i = 0; i < t.size(); ++i)
    oss << t[i] << " ";
  return oss.str();
}

// Return a printed representation of the size of the matrix.
template<typename T>
std::string GetMatrixSize(
    T& matrix,
    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0)
{
  std::ostringstream oss;
  oss << matrix.n_rows << "x" << matrix.n_cols << " matrix";
  return oss.str();
}

// Return a printed representation of the size of the matrix.
template<typename T>
std::string GetMatrixSize(
    T& matrixAndInfo,
    const std::enable_if_t<std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0)
{
  return GetMatrixSize(std::get<1>(matrixAndInfo));
}

//! Print a matrix/tuple option (this just prints the filename).
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<arma::is_arma_type<T>::value || std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* /* junk */)
{
  // Extract the string from the tuple that's being held.
  using TupleType = std::tuple<T, typename ParameterType<T>::type>;
  const TupleType* tuple = std::any_cast<TupleType>(&data.value);

  std::ostringstream oss;
  oss << "'" << std::get<0>(std::get<1>(*tuple)) << "'";

  if (std::get<0>(std::get<1>(*tuple)) != "")
  {
    // Make sure the matrix is loaded so that we can print its size.
    GetParam<T>(const_cast<util::ParamData&>(data));
    std::string matDescription =
        std::to_string(std::get<2>(std::get<1>(*tuple))) + "x" +
        std::to_string(std::get<1>(std::get<1>(*tuple))) + " matrix";

    oss << " (" << matDescription << ")";
  }

  return oss.str();
}

//! Print a model option (this just prints the filename).
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<data::HasSerialize<T>::value>*)
{
  // Extract the string from the tuple that's being held.
  using TupleType = std::tuple<T*, typename ParameterType<T>::type>;
  const TupleType* tuple = std::any_cast<TupleType>(&data.value);

  std::ostringstream oss;
  oss << std::get<1>(*tuple);
  return oss.str();
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
