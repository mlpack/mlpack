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
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  return "unknown";
}

template<>
inline std::string GetGoType<int>(
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
inline std::string GetGoType<float>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<float>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<float>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<float>::value>::type*,
    const typename std::enable_if<!std::is_same<float,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "float32";
}

template<>
inline std::string GetGoType<double>(
    util::ParamData& /* d */,
    const typename std::enable_if<!util::IsStdVector<double>::value>::type*,
    const typename std::enable_if<!data::HasSerialize<double>::value>::type*,
    const typename std::enable_if<!arma::is_arma_type<double>::value>::type*,
    const typename std::enable_if<!std::is_same<double,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type*)
{
  return "float64";
}

template<>
inline std::string GetGoType<std::string>(
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
  return "string";
}

template<>
inline std::string GetGoType<bool>(
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
inline std::string GetGoType(
    util::ParamData& d,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0)
{
  return "[]" + GetGoType<typename T::value_type>(d);
}

template<typename T>
inline std::string GetGoType(
    util::ParamData& /* d */,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0)
{
  return "mat.Dense";
}

template<typename T>
inline std::string GetGoType(
    util::ParamData& /* d */,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0)
{
  return "matrixWithInfo";
}

template<typename T>
inline std::string GetGoType(
    util::ParamData& d,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0)
{
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);
  return goStrippedType;
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
