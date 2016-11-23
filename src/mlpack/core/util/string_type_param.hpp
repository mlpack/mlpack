/**
 * @file string_type_param.hpp
 * @author Ryan Curtin
 *
 * Given a ParamData object, return a string containing the type of the input
 * parameter as given on the command-line.
 */
#ifndef MLPACK_CORE_UTIL_STRING_TYPE_PARAM_HPP
#define MLPACK_CORE_UTIL_STRING_TYPE_PARAM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace util {

/**
 * Return a string containing the type of the parameter.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Return a string containing the type of the parameter, for vector options.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::enable_if<IsStdVector<T>>::type* = 0);

/**
 * Return a string containing the type of the parameter, for matrix options.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);

/**
 * Return a string containing the type of the parameter,
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Return a string containing the type of a parameter.  This overload is used if
 * we don't have a primitive type.
 */
template<typename T>
std::string StringTypeParam()
{
  return StringTypeParamImpl<T>();
}

//! Return "int".
template<>
inline std::string StringTypeParam<int>();

//! Return "bool".
template<>
inline std::string StringTypeParam<bool>();

//! Return "string".
template<>
inline std::string StringTypeParam<std::string>();

//! Return "float".
template<>
inline std::string StringTypeParam<float>();

//! Return "double".
template<>
inline std::string StringTypeParam<double>();

} // namespace util
} // namespace mlpack

// Include implementation.
#include "string_type_param_impl.hpp"

#endif
