/**
 * @file bindings/markdown/get_printable_param_name.hpp
 * @author Ryan Curtin
 *
 * Return the parameter name that the user would specify on the command line,
 * with different behavior for different parameter types.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_PARAM_NAME_HPP
#define MLPACK_BINDINGS_MARKDOWN_GET_PRINTABLE_PARAM_NAME_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Get the parameter name for a type that has no special handling.
 */
template<typename T>
std::string GetPrintableParamName(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0);

/**
 * Get the parameter name for a matrix type (where the user has to pass the file
 * that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamName(
    util::ParamData& data,
    const std::enable_if_t<arma::is_arma_type<T>::value>* = 0);

/**
 * Get the parameter name for a serializable model type (where the user has to
 * pass the file that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamName(
    util::ParamData& data,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<data::HasSerialize<T>::value>* = 0);

/**
 * Get the parameter name for a mapped matrix type (where the user has to pass
 * the file that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamName(
    util::ParamData& data,
    const std::enable_if_t<std::is_same_v<T,
        std::tuple<data::DatasetInfo, arma::mat>>>* = 0);

/**
 * Get the parameter's name as seen by the user.
 */
template<typename T>
void GetPrintableParamName(
    util::ParamData& d,
    const void* /* input */,
    void* output)
{
  *((std::string*) output) =
      GetPrintableParamName<std::remove_pointer_t<T>>(d);
}

} // namespace markdown
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "get_printable_param_name_impl.hpp"

#endif
