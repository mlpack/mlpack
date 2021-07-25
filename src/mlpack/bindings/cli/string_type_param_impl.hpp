/**
 * @file bindings/cli/string_type_param_impl.hpp
 * @author Ryan Curtin
 *
 * Implementations of StringTypeParam().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_STRING_TYPE_PARAM_IMPL_HPP
#define MLPACK_BINDINGS_CLI_STRING_TYPE_PARAM_IMPL_HPP

#include "string_type_param.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Return a string containing the type of the parameter.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* /* junk */,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* /* junk */)
{
  // Don't know what type this is.
  return "unknown";
}

/**
 * Return a string containing the type of the parameter, for vector options.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename std::enable_if<util::IsStdVector<T>::value>::type* /* junk */)
{
  return "vector";
}

/**
 * Return a string containing the type of the parameter,
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename std::enable_if<data::HasSerialize<T>::value>::type* /* junk */)
{
  return "string";
}

//! Return "int".
template<>
inline void StringTypeParam<int>(util::ParamData& /* data */,
                                 const void* /* input */,
                                 void* output)
{
  std::string* outstr = (std::string*) output;
  *outstr = "int";
}

//! Return "bool".
template<>
inline void StringTypeParam<bool>(util::ParamData& /* data */,
                                  const void* /* input */,
                                  void* output)
{
  std::string* outstr = (std::string*) output;
  *outstr = "bool";
}

//! Return "string".
template<>
inline void StringTypeParam<std::string>(util::ParamData& /* data */,
                                         const void* /* input */,
                                         void* output)
{
  std::string* outstr = (std::string*) output;
  *outstr = "string";
}

//! Return "double".
template<>
inline void StringTypeParam<double>(util::ParamData& /* data */,
                                    const void* /* input */,
                                    void* output)
{
  std::string* outstr = (std::string*) output;
  *outstr = "double";
}

//! Return "string";
template<>
inline void StringTypeParam<std::tuple<mlpack::data::DatasetInfo, arma::mat>>(
    util::ParamData& /* data */,
    const void* /* input */,
    void* output)
{
  std::string* outstr = (std::string*) output;
  *outstr = "string";
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
