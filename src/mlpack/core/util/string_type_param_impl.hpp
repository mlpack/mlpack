/**
 * @file string_type_param_impl.hpp
 * @author Ryan Curtin
 *
 * Implementations of StringTypeParam().
 */
#ifndef MLPACK_CORE_UTIL_STRING_TYPE_PARAM_IMPL_HPP
#define MLPACK_CORE_UTIL_STRING_TYPE_PARAM_IMPL_HPP

#include "string_type_param.hpp"

namespace mlpack {
namespace util {

/**
 * Return a string containing the type of the parameter.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<IsStdVector<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */)
{
  // Don't know what type this is.
  return "unknown";
}

/**
 * Return a string containing the type of the parameter, for vector options.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::enable_if<IsStdVector<T>>::type* /* junk */)
{
  return "vector";
}

/**
 * Return a string containing the type of the parameter, for matrix options.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::enable_if<arma::is_arma_type<T>>::type* /* junk */)
{
  return "string";
}

/**
 * Return a string containing the type of the parameter,
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::enable_if<data::HasSerialize<T>>::type* /* junk */)
{
  return "string";
}

//! Return "int".
template<>
inline std::string StringTypeParam<int>()
{
  return "int";
}

//! Return "bool".
template<>
inline std::string StringTypeParam<bool>()
{
  return "bool";
}

//! Return "string".
template<>
inline std::string StringTypeParam<std::string>()
{
  return "string";
}

//! Return "float".
template<>
inline std::string StringTypeParam<float>()
{
  return "float";
}

//! Return "double".
template<>
inline std::string StringTypeParam<double>()
{
  return "double";
}

//! Return "string";
template<>
inline std::string StringTypeParam<std::tuple<mlpack::data::DatasetInfo,
    arma::mat>>()
{
  return "string";
}

} // namespace util
} // namespace mlpack

#endif
