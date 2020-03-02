/**
 * @file get_printable_type_impl.hpp
 * @author Yashwant Singh
 *
 * Template metaprogramming to return the string representation of the Go
 * type for a given Go binding parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_GET_PRINTABLE_TYPE_IMPL_HPP
#define MLPACK_BINDINGS_GO_GET_PRINTABLE_TYPE_IMPL_HPP

#include "get_printable_type.hpp"

namespace mlpack {
namespace bindings {
namespace go {

template<typename T>
inline std::string GetPrintableType(
    const util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<T>>::type*,
    const typename boost::disable_if<data::HasSerialize<T>>::type*,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "unknown";
}

template<>
inline std::string GetPrintableType<int>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<int>>::type*,
    const typename boost::disable_if<data::HasSerialize<int>>::type*,
    const typename boost::disable_if<arma::is_arma_type<int>>::type*,
    const typename boost::disable_if<std::is_same<int,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "int";
}

template<>
inline std::string GetPrintableType<double>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<double>>::type*,
    const typename boost::disable_if<data::HasSerialize<double>>::type*,
    const typename boost::disable_if<arma::is_arma_type<double>>::type*,
    const typename boost::disable_if<std::is_same<double,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "float64";
}

template<>
inline std::string GetPrintableType<std::string>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<std::string>>::type*,
    const typename boost::disable_if<data::HasSerialize<std::string>>::type*,
    const typename boost::disable_if<arma::is_arma_type<std::string>>::type*,
    const typename boost::disable_if<std::is_same<std::string,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "string";
}

template<>
inline std::string GetPrintableType<bool>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<bool>>::type*,
    const typename boost::disable_if<data::HasSerialize<bool>>::type*,
    const typename boost::disable_if<arma::is_arma_type<bool>>::type*,
    const typename boost::disable_if<std::is_same<bool,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "bool";
}

template<typename T>
inline std::string GetPrintableType(
    const util::ParamData& d,
    const typename boost::enable_if<util::IsStdVector<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "array of " + GetPrintableType<typename T::value_type>(d) + "s";
}

template<typename T>
inline std::string GetPrintableType(
    const util::ParamData& /* d */,
    const typename boost::enable_if<arma::is_arma_type<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  std::string type = "*mat.Dense";
  if (std::is_same<typename T::elem_type, double>::value)
  {
    if (T::is_row || T::is_col)
      type = "*mat.Dense (1d)";
  }
  else if (std::is_same<typename T::elem_type, size_t>::value)
  {
    type = "*mat.Dense (with ints)";
    if (T::is_row || T::is_col)
      type = "*mat.Dense (1d with ints)";
  }

  return type;
}

template<typename T>
inline std::string GetPrintableType(
    const util::ParamData& /* d */,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "matrixWithInfo";
}

template<typename T>
inline std::string GetPrintableType(
    const util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::enable_if<data::HasSerialize<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return d.cppType + "Type";
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
