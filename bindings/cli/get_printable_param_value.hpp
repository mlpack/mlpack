/**
 * @file bindings/cli/get_printable_param_value.hpp
 * @author Ryan Curtin
 *
 * Given a parameter value, print what the user might actually specify on the
 * command line.  Basically this adds ".csv" to types where data must be loaded.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_VALUE_HPP
#define MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_VALUE_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Get the parameter name for a type that has no special handling.
 */
template<typename T>
std::string GetPrintableParamValue(
    util::ParamData& data,
    const std::string& value,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<!data::HasSerialize<T>::value>::type* = 0,
    const typename std::enable_if<!std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Get the parameter name for a matrix type (where the user has to pass the file
 * that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamValue(
    util::ParamData& data,
    const std::string& value,
    const typename std::enable_if<arma::is_arma_type<T>::value>::type* = 0);

/**
 * Get the parameter name for a serializable model type (where the user has to
 * pass the file that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamValue(
    util::ParamData& data,
    const std::string& value,
    const typename std::enable_if<!arma::is_arma_type<T>::value>::type* = 0,
    const typename std::enable_if<data::HasSerialize<T>::value>::type* = 0);

/**
 * Get the parameter name for a mapped matrix type (where the user has to pass
 * the file that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamValue(
    util::ParamData& data,
    const std::string& value,
    const typename std::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>::value>::type* = 0);

/**
 * Get the parameter's name as seen by the user.
 */
template<typename T>
void GetPrintableParamValue(
    util::ParamData& d,
    const void* input,
    void* output)
{
  *((std::string*) output) =
      GetPrintableParamValue<typename std::remove_pointer<T>::type>(d,
      *((std::string*) input));
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "get_printable_param_value_impl.hpp"

#endif
