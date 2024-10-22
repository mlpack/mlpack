/**
 * @file bindings/tests/get_printable_param_impl.hpp
 * @author Ryan Curtin
 *
 * Implementation of parameter printing functions.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_TESTS_GET_PRINTABLE_PARAM_IMPL_HPP
#define MLPACK_BINDINGS_TESTS_GET_PRINTABLE_PARAM_IMPL_HPP

#include "get_printable_param.hpp"

namespace mlpack {
namespace bindings {
namespace tests {

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
    const std::enable_if_t<util::IsStdVector<T>::value>*)
{
  const T& t = std::any_cast<T>(data.value);

  std::ostringstream oss;
  for (size_t i = 0; i < t.size(); ++i)
    oss << t[i] << " ";
  return oss.str();
}

//! Print a matrix option (this just prints 'matrix type').
template<typename T>
std::string GetPrintableParam(
    util::ParamData& /* data */,
    const std::enable_if_t<arma::is_arma_type<T>::value>*)
{
  return "matrix type";
}

//! Print a model option (this just prints the filename).
template<typename T>
std::string GetPrintableParam(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>*,
    const std::enable_if_t<data::HasSerialize<T>::value>*)
{
  // Extract the string from the tuple that's being held.
  std::ostringstream oss;
  oss << data.cppType << " model";
  return oss.str();
}

//! Print a mapped matrix option (this just prints 'matrix/DatasetInfo tuple').
template<typename T>
std::string GetPrintableParam(
    util::ParamData& /* data */,
    const std::enable_if_t<std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* /* junk */)
{
  return "matrix/DatatsetInfo tuple";
}

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
