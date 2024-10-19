/**
 * @file bindings/go/get_type.hpp
 * @author Yasmine Dumouchel
 *
 * Template metaprogramming to return the string representation of the type
 * for the Go bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_GET_TYPE_HPP
#define MLPACK_BINDINGS_GO_GET_TYPE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/param_data.hpp>
#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace go {

template<typename T>
inline std::string GetType(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0)
{
  return "unknown";
}

template<>
inline std::string GetType<int>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<int>::value>*,
    const std::enable_if_t<!data::HasSerialize<int>::value>*,
    const std::enable_if_t<!arma::is_arma_type<int>::value>*)
{
  return "Int";
}

template<>
inline std::string GetType<float>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<float>::value>*,
    const std::enable_if_t<!data::HasSerialize<float>::value>*,
    const std::enable_if_t<!arma::is_arma_type<float>::value>*)
{
  return "Float";
}

template<>
inline std::string GetType<double>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<double>::value>*,
    const std::enable_if_t<!data::HasSerialize<double>::value>*,
    const std::enable_if_t<!arma::is_arma_type<double>::value>*)
{
  return "Double";
}

template<>
inline std::string GetType<std::string>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<std::string>::value>*,
    const std::enable_if_t<!data::HasSerialize<std::string>::value>*,
    const std::enable_if_t<!arma::is_arma_type<std::string>::value>*)
{
  return "String";
}

template<>
inline std::string GetType<bool>(
    util::ParamData& /* d */,
    const std::enable_if_t<!util::IsStdVector<bool>::value>*,
    const std::enable_if_t<!data::HasSerialize<bool>::value>*,
    const std::enable_if_t<!arma::is_arma_type<bool>::value>*)
{
  return "Bool";
}

template<typename T>
inline std::string GetType(
    util::ParamData& d,
    const std::enable_if_t<util::IsStdVector<T>::value>* = 0)
{
  return "Vec" + GetType<typename T::value_type>(d);
}

template<typename T>
inline std::string GetType(
    util::ParamData& /* d */,
    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0)
{
  std::string type = "";
  if (std::is_same_v<typename T::elem_type, double>)
  {
    if (T::is_row)
      type = "Row";
    else if (T::is_col)
      type = "Col";
    else
      type = "Mat";
  }
  else if (std::is_same_v<typename T::elem_type, size_t>)
  {
    if (T::is_row)
      type = "Urow";
    else if (T::is_col)
      type = "Ucol";
    else
      type = "Umat";
  }

  return type;
}

template<typename T>
inline std::string GetType(
    util::ParamData& d,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0)
{
  return d.cppType + "*";
}

/**
 * Function is used to generate the type names that are used in calls to
 * functions like gonumToArma<type>() or setParam<type>(), and therefore
 * what's returned isn't exactly the Go native type used for that parameter
 * type.
 *
 * @param d Parameter data struct.
 * @param * (input) Unused parameter.
 * @param output Output storage for the string.
 */
template<typename T>
void GetType(util::ParamData& d,
             const void* /* input */,
             void* output)
{
  *((std::string*) output) = GetType<std::remove_pointer_t<T>>(d);
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
