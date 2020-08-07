/**
 * @file bindings/R/default_param.hpp
 * @author Yashwant Singh Parihar
 *
 * Return the default value of a parameter, depending on its type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_DEFAULT_PARAM_HPP
#define MLPACK_BINDINGS_R_DEFAULT_PARAM_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/param_data.hpp>
#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Return the default value of an option.  This is for regular types.
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T, std::string>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0);

/**
 * Return the default value of a vector option.
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& data,
    const typename boost::enable_if<util::IsStdVector<T>>::type* = 0);

/**
 * Return the default value of a string option.
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& data,
    const typename boost::enable_if<std::is_same<T, std::string>>::type* = 0);

/**
 * Return the default value of a matrix option, a tuple option, a
 * serializable option, or a string option (this returns the default filename,
 * or '' if the default is no file).
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& data,
    const typename boost::enable_if_c<
        arma::is_arma_type<T>::value ||
        std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                   arma::mat>>::value>::type* /* junk */ = 0);

/**
 * Return the default value of a model option (this returns the default
 * filename, or '' if the default is no file).
 */
template<typename T>
std::string DefaultParamImpl(
    util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Return the default value of an option.  This is the function that will be
 * placed into the IO functionMap.
 */
template<typename T>
void DefaultParam(util::ParamData& data,
                  const void* /* input */,
                  void* output)
{
  std::string* outstr = (std::string*) output;
  *outstr = DefaultParamImpl<typename std::remove_pointer<T>::type>(data);
}

} // namespace r
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "default_param_impl.hpp"

#endif
