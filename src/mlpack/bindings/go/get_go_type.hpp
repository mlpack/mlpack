/**
 * @file get_go_type.hpp
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
    const util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  return "unknown";
}

template<>
inline std::string GetGoType<int>(
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
inline std::string GetGoType<float>(
    const util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<float>>::type*,
    const typename boost::disable_if<data::HasSerialize<float>>::type*,
    const typename boost::disable_if<arma::is_arma_type<float>>::type*,
    const typename boost::disable_if<std::is_same<float,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "float32";
}

template<>
inline std::string GetGoType<double>(
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
inline std::string GetGoType<std::string>(
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
inline std::string GetGoType<bool>(
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
inline std::string GetGoType(
    const util::ParamData& d,
    const typename boost::enable_if<util::IsStdVector<T>>::type* = 0)
{
  return "[]" + GetGoType<typename T::value_type>(d);
}

template<typename T>
inline std::string GetGoType(
    const util::ParamData& /* d */,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
{
  return "mat.Dense";
}

template<typename T>
inline std::string GetGoType(
    const util::ParamData& /* d */,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  return "matrixWithInfo";
}

template<typename T>
inline std::string GetGoType(
    const util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(d.cppType, goStrippedType, strippedType, printedType, defaultsType);
  return goStrippedType;
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
