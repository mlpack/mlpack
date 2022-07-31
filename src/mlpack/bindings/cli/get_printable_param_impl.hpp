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
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* /* junk */,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* /* junk */,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* /* junk */,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* /* junk */)
{
  std::ostringstream oss;
  oss << MLPACK_ANY_CAST<T>(data.value);
  return oss.str();
}

//! Print a vector option.
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type*
        /* junk */)
{
  const T& t = MLPACK_ANY_CAST<T>(data.value);

  std::ostringstream oss;
  for (size_t i = 0; i < t.size(); ++i)
    oss << t[i] << " ";
  return oss.str();
}

// Return a printed representation of the size of the matrix.
template<typename T>
std::string GetMatrixSize(
    T& matrix,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  std::ostringstream oss;
  oss << matrix.n_rows << "x" << matrix.n_cols << " matrix";
  return oss.str();
}

// Return a printed representation of the size of the matrix.
template<typename T>
std::string GetMatrixSize(
    T& matrixAndInfo,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  return GetMatrixSize(std::get<1>(matrixAndInfo));
}

//! Print a matrix/tuple option (this just prints the filename).
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value ||
                                  std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* /* junk */)
{
  // Extract the string from the tuple that's being held.
  typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
  const TupleType* tuple = MLPACK_ANY_CAST<TupleType>(&data.value);

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
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* /* junk */,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* /* junk */)
{
  // Extract the string from the tuple that's being held.
  typedef std::tuple<T*, typename ParameterType<T>::type> TupleType;
  const TupleType* tuple = MLPACK_ANY_CAST<TupleType>(&data.value);

  std::ostringstream oss;
  oss << std::get<1>(*tuple);
  return oss.str();
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
