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
    const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  return "unknown";
}

template<>
inline std::string GetRType<bool>(
    util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<bool>>::type*,
    const typename boost::disable_if<data::HasSerialize<bool>>::type*,
    const typename boost::disable_if<arma::is_arma_type<bool>>::type*,
    const typename boost::disable_if<std::is_same<bool,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "logical";
}

template<>
inline std::string GetRType<int>(
    util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<int>>::type*,
    const typename boost::disable_if<data::HasSerialize<int>>::type*,
    const typename boost::disable_if<arma::is_arma_type<int>>::type*,
    const typename boost::disable_if<std::is_same<int,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "integer";
}

template<>
inline std::string GetRType<size_t>(
    util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<size_t>>::type*,
    const typename boost::disable_if<data::HasSerialize<size_t>>::type*,
    const typename boost::disable_if<arma::is_arma_type<size_t>>::type*,
    const typename boost::disable_if<std::is_same<size_t,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "integer";
}

template<>
inline std::string GetRType<double>(
    util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<double>>::type*,
    const typename boost::disable_if<data::HasSerialize<double>>::type*,
    const typename boost::disable_if<arma::is_arma_type<double>>::type*,
    const typename boost::disable_if<std::is_same<double,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "numeric";
}

template<>
inline std::string GetRType<std::string>(
    util::ParamData& /* d */,
    const typename boost::disable_if<util::IsStdVector<std::string>>::type*,
    const typename boost::disable_if<data::HasSerialize<std::string>>::type*,
    const typename boost::disable_if<arma::is_arma_type<std::string>>::type*,
    const typename boost::disable_if<std::is_same<std::string,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return "character";
}

template<typename T>
inline std::string GetRType(
    util::ParamData& d,
    const typename boost::enable_if<util::IsStdVector<T>>::type* = 0)
{
  return GetRType<typename T::value_type>(d) + " vector";
}

template<typename T>
inline std::string GetRType(
    util::ParamData& d,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0)
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
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0)
{
  return "numeric matrix/data.frame with info";
}

template<typename T>
inline std::string GetRType(
    util::ParamData& d,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0)
{
  return util::StripType(d.cppType);
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
