/**
 * @file get_printable_param_impl.hpp
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

namespace mlpack {
namespace bindings {
namespace cli {

//! Print an option.
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<util::IsStdVector<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  std::ostringstream oss;
  oss << boost::any_cast<T>(data.value);
  return oss.str();
}

//! Print a vector option.
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type*
        /* junk */)
{
  const T& t = boost::any_cast<T>(data.value);

  std::ostringstream oss;
  for (size_t i = 0; i < t.size(); ++i)
    oss << t[i] << " ";
  return oss.str();
}

//! Print a matrix/tuple option (this just prints the filename).
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value ||
                                  std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* /* junk */)
{
  // Extract the string from the tuple that's being held.
  typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
  const TupleType* tuple = boost::any_cast<TupleType>(&data.value);

  std::ostringstream oss;
  oss << std::get<1>(*tuple);
  return oss.str();
}

//! Print a model option (this just prints the filename).
template<typename T>
std::string GetPrintableParam(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
{
  // Extract the string from the tuple that's being held.
  typedef std::tuple<T*, typename ParameterType<T>::type> TupleType;
  const TupleType* tuple = boost::any_cast<TupleType>(&data.value);

  std::ostringstream oss;
  oss << std::get<1>(*tuple);
  return oss.str();
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
