/**
 * @file print_param.hpp
 * @author Ryan Curtin
 *
 * Print the parameter to stdout, using template metaprogramming to enforce
 * different behavior for different parameter types.
 */
#ifndef MLPACK_CORE_UTIL_PRINT_PARAM_HPP
#define MLPACK_CORE_UTIL_PRINT_PARAM_HPP

#include <mlpack/prereqs.hpp>
#include "param_data.hpp"

namespace mlpack {
namespace util {

/**
 * Print an option.
 */
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Print a vector option, with spaces between it.
 */
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::enable_if<IsStdVector<T>>::type* = 0);

/**
 * Print a matrix option (this just prints the filename).
 */
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);

/**
 * Print a serializable class option (this just prints the filename).
 */
template<typename T>
void PrintParamImpl(
    const ParamData& data,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Print an option.
 */
template<typename T>
void PrintParam(const ParamData& data)
{
  PrintParamImpl<T>(data);
}

} // namespace util
} // namespace mlpack

// Include implementation.
#include "print_param_impl.hpp"

#endif
