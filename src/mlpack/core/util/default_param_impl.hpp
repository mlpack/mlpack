/**
 * @file default_param_impl.hpp
 * @author Ryan Curtin
 *
 * Return the default value of a parameter, depending on its type.
 */
#ifndef MLPACK_CORE_UTIL_DEFAULT_PARAM_IMPL_HPP
#define MLPACK_CORE_UTIL_DEFAULT_PARAM_IMPL_HPP

#include "default_param.hpp"

namespace mlpack {
namespace util {

/**
 * Return the default value of an option.
 */
template<typename T>
std::string DefaultParamImpl(
    const ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* /* junk */,
    const typename boost::disable_if<IsStdVector<T>>::type* /* junk */,
    const typename boost::disable_if<data::HasSerialize<T>>::type* /* junk */,
    const typename boost::disable_if<std::is_same<T, std::string>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* /* junk */)
{
  std::ostringstream oss;
  oss << boost::any_cast<T>(data.value);
  return oss.str();
}

/**
 * Return the default value of a vector option.
 */
template<typename T>
std::string DefaultParamImpl(
    const ParamData& data,
    const typename boost::enable_if<IsStdVector<T>>::type* /* junk */)
{
  // Print each element in an array delimited by square brackets.
  std::ostringstream oss;
  const T& vector = boost::any_cast<T>(data.value);
  oss << "[";
  for (size_t i = 0; i < vector.size() - 1; ++i)
    oss << vector[i] << " ";
  oss << vector[vector.size() - 1] << "]";
  return oss.str();
}

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
        std::is_same<T, std::string>::value>::type* /* junk */)
{
  // Get the filename and return it, or return an empty string.
  const std::string& filename = boost::any_cast<std::string>(data.value);
  return "'" + filename + "'";
}

} // namespace util
} // namespace mlpack

#endif
