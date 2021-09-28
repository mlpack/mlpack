/**
 * @file bindings/python/get_printable_type_impl.hpp
 * @author Ryan Curtin
 *
 * Template metaprogramming to return the string representation of the Python
 * type for a given Python binding parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GET_PRINTABLE_TYPE_IMPL_HPP
#define MLPACK_BINDINGS_PYTHON_GET_PRINTABLE_TYPE_IMPL_HPP

#include "get_printable_type.hpp"

namespace mlpack {
namespace bindings {
namespace python {

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "unknown";
}

template<>
inline std::string GetPrintableType<int>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<int>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<int>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<int>::value>::type*,
    const typename std::enable_if<!std::is_same<int,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "int";
}

template<>
inline std::string GetPrintableType<double>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<double>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<double>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<double>::value>::type*,
    const typename std::enable_if<!std::is_same<double,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "float";
}

template<>
inline std::string GetPrintableType<std::string>(
    util::ParamData& /* d */,
    const typename std::enable_if<
        !util::IsStdVector<std::string>::value>::type*,
    const typename std::enable_if<
        !data::HasSerialize<std::string>::value>::type*,
    const typename std::enable_if<
        !arma::is_arma_type<std::string>::value>::type*,
    const typename std::enable_if<!std::is_same<std::string,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "str";
}

template<>
inline std::string GetPrintableType<size_t>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<size_t>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<size_t>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<size_t>::value>::type*,
    const typename std::enable_if<!std::is_same<size_t,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "int";
}

template<>
inline std::string GetPrintableType<bool>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<bool>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<bool>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<bool>::value>::type*,
    const typename std::enable_if<!std::is_same<bool,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "bool";
}

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& d,
    const typename std::enable_if<util::IsStdVector<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "list of " + GetPrintableType<typename T::value_type>(d) + "s";
}

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& /* d */,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  std::string type = "matrix";
  if (std::is_same<typename T::elem_type, double>::value)
  {
    if (T::is_row || T::is_col)
      type = "vector";
  }
  else if (std::is_same<typename T::elem_type, size_t>::value)
  {
    type = "int matrix";
    if (T::is_row || T::is_col)
      type = "int vector";
  }

  return type;
}

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& /* d */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "categorical matrix";
}

template<typename T>
inline std::string GetPrintableType(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type*,
    const typename std::enable_if<data::HasSerialize<T>::value>::type*,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return d.cppType + "Type";
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
