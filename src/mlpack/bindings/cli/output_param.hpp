/**
 * @file bindings/cli/output_param.hpp
 * @author Ryan Curtin
 *
 * Output a parameter of different types using template metaprogramming.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_OUTPUT_PARAM_HPP
#define MLPACK_BINDINGS_CLI_OUTPUT_PARAM_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/param_data.hpp>
#include <mlpack/core/util/is_std_vector.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Output an option (print to stdout).
 */
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!util::IsStdVector<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Output a vector option (print to stdout).
 */
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const typename std::enable_if<util::IsStdVector<T>::value>::type* = 0);

/**
 * Output a matrix option (this saves it to the given file).
 */
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0);

/**
 * Output a serializable class option (this saves it to the given file).
 */
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0);

/**
 * Output a mapped dataset.
 */
template<typename T>
void OutputParamImpl(
    util::ParamData& data,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Output an option.  This is the function that will be called by the IO
 * module.
 */
template<typename T>
void OutputParam(util::ParamData& data,
                 const void* /* input */,
                 void* /* output */)
{
  OutputParamImpl<typename std::remove_pointer<T>::type>(data);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "output_param_impl.hpp"

#endif
