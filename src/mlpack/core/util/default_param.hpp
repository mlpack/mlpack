/**
 * @file default_param.hpp
 * @author Ryan Curtin
 *
 * Return the default value of a parameter, depending on its type.
 */
#ifndef MLPACK_CORE_UTIL_DEFAULT_PARAM_HPP
#define MLPACK_CORE_UTIL_DEFAULT_PARAM_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace util {

/**
 * Return the default value of an option.
 */
template<typename T>
std::string DefaultParamImpl(
    const ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T, std::string>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0);

/**
 * Return the default value of a vector option.
 */
template<typename T>
std::string DefaultParamImpl(
    const ParamData& data,
    const typename boost::enable_if<IsStdVector<T>>::type* = 0);

/**
 * Return the default value of a matrix option (this returns the default
 * filename, or '' if the default is no file).
 */
template<typename T>
std::string DefaultParamImpl(
    const ParamData& data,
    const typename boost::enable_if_c<
        arma::is_arma_type<T>::value ||
        data::HasSerialize<T>::value ||
        std::is_same<T, std::tuple<mlpack::data::DatasetInfo,
                                   arma::mat>>::value ||
        std::is_same<T, std::string>::value>::type* /* junk */ = 0);

/**
 * Return the default value of an option.  This is the function that will be
 * called by the CLI module.
 */
template<typename T>
std::string DefaultParam(const ParamData& data)
{
  return DefaultParamImpl<T>(data);
}

} // namespace util
} // namespace mlpack

// Include implementation.
#include "default_param_impl.hpp"

#endif
