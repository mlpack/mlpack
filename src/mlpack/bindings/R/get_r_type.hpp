/**
 * @file bindings/R/get_r_type.hpp
 * @author Yashwant Singh Parihar
 *
 * Get the R-named type of an mlpack C++ type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_GET_R_TYPE_HPP
#define MLPACK_BINDINGS_R_GET_R_TYPE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/is_std_vector.hpp>
#include <mlpack/bindings/util/strip_type.hpp>

namespace mlpack {
namespace bindings {
namespace r {

template<typename T>
inline std::string GetRType(
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
inline std::string GetRType<bool>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<bool>::value>*,
    const std::enable_if_t<!data::HasSerialize<bool>::value>*,
    const std::enable_if_t<!arma::is_arma_type<bool>::value>*,
    const std::enable_if_t<!std::is_same_v<bool,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "logical";
}

template<>
inline std::string GetRType<int>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<int>::value>*,
    const std::enable_if_t<!data::HasSerialize<int>::value>*,
    const std::enable_if_t<!arma::is_arma_type<int>::value>*,
    const std::enable_if_t<!std::is_same_v<int,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "integer";
}

template<>
inline std::string GetRType<size_t>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<size_t>::value>*,
    const std::enable_if_t<!data::HasSerialize<size_t>::value>*,
    const std::enable_if_t<!arma::is_arma_type<size_t>::value>*,
    const std::enable_if_t<!std::is_same_v<size_t,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "integer";
}

template<>
inline std::string GetRType<double>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<double>::value>*,
    const std::enable_if_t<!data::HasSerialize<double>::value>*,
    const std::enable_if_t<!arma::is_arma_type<double>::value>*,
    const std::enable_if_t<!std::is_same_v<double,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "numeric";
}

template<>
inline std::string GetRType<std::string>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<std::string>::value>*,
    const std::enable_if_t<!data::HasSerialize<std::string>::value>*,
    const std::enable_if_t<!arma::is_arma_type<std::string>::value>*,
    const std::enable_if_t<
        !std::is_same_v<std::string,
        std::tuple<data::DatasetInfo, arma::mat>>>*)
{
  return "character";
}

template<typename T>
inline std::string GetRType(
    util::ParamData& d,
    const std::enable_if_t<util::IsStdVector<T>::value>* = 0)
{
  return GetRType<typename T::value_type>(d) + " vector";
}

template<typename T>
inline std::string GetRType(
    util::ParamData& d,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0,
    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0)
{
  std::string elemType = GetRType<typename T::elem_type>(d);
  std::string type = "matrix";
  if (T::is_row)
    type = "row";
  else if (T::is_col)
    type = "column";

  return  elemType + " " + type;
}

template<typename T>
inline std::string GetRType(
    util::ParamData& /* d */,
    const std::enable_if_t<std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0)
{
  return "numeric matrix/data.frame with info";
}

template<typename T>
inline std::string GetRType(
    util::ParamData& d,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  return util::StripType(d.cppType);
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
