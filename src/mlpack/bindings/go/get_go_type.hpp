/**
 * @file bindings/go/get_go_type.hpp
 * @author Yasmine Dumouchel
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
#ifndef MLPACK_BINDINGS_GO_GET_GO_TYPE_HPP
#define MLPACK_BINDINGS_GO_GET_GO_TYPE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/is_std_vector.hpp>
#include "strip_type.hpp"

namespace mlpack {
namespace bindings {
namespace go {

template<typename T>
inline std::string GetGoType(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0)
{
  return "unknown";
}

template<>
inline std::string GetGoType<int>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<int>::value>*,
    const std::enable_if_t<!data::HasSerialize<int>::value>*,
    const std::enable_if_t<!arma::is_arma_type<int>::value>*,
    const std::enable_if_t<!std::is_same_v<int,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "int";
}

template<>
inline std::string GetGoType<float>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<float>::value>*,
    const std::enable_if_t<!data::HasSerialize<float>::value>*,
    const std::enable_if_t<!arma::is_arma_type<float>::value>*,
    const std::enable_if_t<!std::is_same_v<float,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "float32";
}

template<>
inline std::string GetGoType<double>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<double>::value>*,
    const std::enable_if_t<!data::HasSerialize<double>::value>*,
    const std::enable_if_t<!arma::is_arma_type<double>::value>*,
    const std::enable_if_t<!std::is_same_v<double,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "float64";
}

template<>
inline std::string GetGoType<std::string>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<std::string>::value>*,
    const std::enable_if_t<!data::HasSerialize<std::string>::value>*,
    const std::enable_if_t<!arma::is_arma_type<std::string>::value>*,
    const std::enable_if_t<!std::is_same_v<std::string,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "string";
}

template<>
inline std::string GetGoType<bool>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<bool>::value>*,
    const std::enable_if_t<!data::HasSerialize<bool>::value>*,
    const std::enable_if_t<!arma::is_arma_type<bool>::value>*,
    const std::enable_if_t<!std::is_same_v<bool,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "bool";
}

template<typename T>
inline std::string GetGoType(
    util::ParamData& d,
    const std::enable_if_t<util::IsStdVector<T>::value>* = 0)
{
  return "[]" + GetGoType<typename T::value_type>(d);
}

template<typename T>
inline std::string GetGoType(
    util::ParamData& /* d */,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0,
    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0)
{
  return "mat.Dense";
}

template<typename T>
inline std::string GetGoType(
    util::ParamData& /* d */,
    const std::enable_if_t<std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0)
{
  return "matrixWithInfo";
}

template<typename T>
inline std::string GetGoType(
    util::ParamData& d,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);
  return goStrippedType;
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
