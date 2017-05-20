/**
 * @file output_param.hpp
 * @author Ryan Curtin
 *
 * Output a parameter of different types using template metaprogramming.
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
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);

/**
 * Output a vector option (print to stdout).
 */
template<typename T>
void OutputParamImpl(
    const util::ParamData& data,
    const typename boost::enable_if<util::IsStdVector<T>>::type* = 0);

/**
 * Output a matrix option (this saves it to the given file).
 */
template<typename T>
void OutputParamImpl(
    const util::ParamData& data,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);

/**
 * Output a serializable class option (this saves it to the given file).
 */
template<typename T>
void OutputParamImpl(
    const util::ParamData& data,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Output a mapped dataset.
 */
template<typename T>
void OutputParamImpl(
    const util::ParamData& data,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);

/**
 * Output an option.  This is the function that will be called by the CLI
 * module.
 */
template<typename T>
void OutputParam(const util::ParamData& data,
                 const void* /* input */,
                 void* /* output */)
{
  OutputParamImpl<T>(data);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "output_param_impl.hpp"

#endif
