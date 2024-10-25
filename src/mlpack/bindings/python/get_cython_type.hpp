/**
 * @file bindings/python/get_cython_type.hpp
 * @author Ryan Curtin
 *
 * Template metaprogramming to return the string representation of the Cython
 * type for a given Cython binding parameter.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_GET_CYTHON_TYPE_HPP
#define MLPACK_BINDINGS_PYTHON_GET_CYTHON_TYPE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace python {

template<typename T>
inline std::string GetCythonType(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0)
{
  return "unknown";
}

template<>
inline std::string GetCythonType<int>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<int>::value>*,
    const std::enable_if_t<!data::HasSerialize<int>::value>*,
    const std::enable_if_t<!arma::is_arma_type<int>::value>*)
{
  return "int";
}

template<>
inline std::string GetCythonType<double>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<double>::value>*,
    const std::enable_if_t<!data::HasSerialize<double>::value>*,
    const std::enable_if_t<!arma::is_arma_type<double>::value>*)
{
  return "double";
}

template<>
inline std::string GetCythonType<std::string>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<std::string>::value>*,
    const std::enable_if_t<!data::HasSerialize<std::string>::value>*,
    const std::enable_if_t<!arma::is_arma_type<std::string>::value>*)
{
  return "string";
}

template<>
inline std::string GetCythonType<size_t>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<size_t>::value>*,
    const std::enable_if_t<!data::HasSerialize<size_t>::value>*,
    const std::enable_if_t<!arma::is_arma_type<size_t>::value>*)
{
  return "size_t";
}

template<>
inline std::string GetCythonType<bool>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<bool>::value>*,
    const std::enable_if_t<!data::HasSerialize<bool>::value>*,
    const std::enable_if_t<!arma::is_arma_type<bool>::value>*)
{
  return "cbool";
}

template<typename T>
inline std::string GetCythonType(
    util::ParamData& d,
    const std::enable_if_t<util::IsStdVector<T>::value>* = 0)
{
  return "vector[" + GetCythonType<typename T::value_type>(d) + "]";
}

template<typename T>
inline std::string GetCythonType(
    util::ParamData& d,
    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0)
{
  std::string type = "Mat";
  if (T::is_row)
    type = "Row";
  else if (T::is_col)
    type = "Col";

  return type + "[" + GetCythonType<typename T::elem_type>(d) + "]";
}

template<typename T>
inline std::string GetCythonType(
    util::ParamData& d,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  return d.cppType + "*";
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
