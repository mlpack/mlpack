/**
 * @file string_type_param.hpp
 * @author Ryan Curtin
 *
 * Given a util::ParamData object, return a string containing the type of the input
 * parameter as given on the command-line.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CMD_STRING_TYPE_PARAM_HPP
#define MLPACK_BINDINGS_CMD_STRING_TYPE_PARAM_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/param_data.hpp>
#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace cmd {

/**
 * Return a string containing the type of the parameter.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Return a string containing the type of the parameter, for vector options.
 */
template<typename T>
std::string StringTypeParamImpl(
    const typename boost::enable_if<util::IsStdVector<T>>::type* = 0);

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
void StringTypeParam(util::ParamData& /* data */,
                     const void* /* input */,
                     void* output)
{
  std::string* outstr = (std::string*) output;
  *outstr = StringTypeParamImpl<T>();
}

//! Return "int".
template<>
inline void StringTypeParam<int>(util::ParamData& /* data */,
                                 const void* /* input */,
                                 void* output);

//! Return "bool".
template<>
inline void StringTypeParam<bool>(util::ParamData& /* data */,
                                  const void* /* input */,
                                  void* output);

//! Return "string".
template<>
inline void StringTypeParam<std::string>(util::ParamData& /* data */,
                                         const void* /* input */,
                                         void* output);

//! Return "double".
template<>
inline void StringTypeParam<double>(util::ParamData& /* data */,
                                    const void* /* input */,
                                    void* output);

//! Return "string";
template<>
inline void StringTypeParam<std::tuple<mlpack::data::DatasetInfo, arma::mat>>(
    util::ParamData& /* data */,
    const void* /* input */,
    void* output);

} // namespace cmd
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "string_type_param_impl.hpp"

#endif
