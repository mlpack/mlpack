/**
 * @file get_printable_param_name.hpp
 * @author Ryan Curtin
 *
 * Return the parameter name that the user would specify on the command line,
 * with different behavior for different parameter types.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_NAME_HPP
#define MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_NAME_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Get the parameter name for a type that has no special handling.
 */
template<typename T>
std::string GetPrintableParamName(
    const util::ParamData& data,
    const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
    const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);

/**
 * Get the parameter name for a matrix type (where the user has to pass the file
 * that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamName(
    const util::ParamData& data,
    const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0);

/**
 * Get the parameter name for a serializable model type (where the user has to
 * pass the file that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamName(
    const util::ParamData& data,
    const typename boost::enable_if<data::HasSerialize<T>>::type* = 0);

/**
 * Get the parameter name for a mapped matrix type (where the user has to pass
 * the file that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamName(
    const util::ParamData& data,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type* = 0);

/**
 * Get the parameter's name as seen by the user.
 */
template<typename T>
void GetPrintableParamName(
    const util::ParamData& d,
    const void* /* input */,
    void* output)
{
  *((std::string*) output) = GetPrintableParamName<T>(d);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "get_printable_param_name_impl.hpp"

#endif
