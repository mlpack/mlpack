/**
 * @file get_printable_param_value_impl.hpp
 * @author Ryan Curtin
 *
 * Return the parameter value that the user would specify on the command line
 * depending on the type of the option.  Basically this adds ".csv" to types
 * that need to be loaded.
 */
#ifndef MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_VALUE_IMPL_HPP
#define MLPACK_BINDINGS_CLI_GET_PRINTABLE_PARAM_VALUE_IMPL_HPP

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
    const util::ParamData& /* data */,
    const std::string& input,
    const typename boost::disable_if<arma::is_arma_type<T>>::type*,
    const typename boost::disable_if<data::HasSerialize<T>>::type*,
    const typename boost::disable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return input;
}

/**
 * Get the parameter name for a matrix type (where the user has to pass the file
 * that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamValue(
    const util::ParamData& /* data */,
    const std::string& input,
    const typename boost::enable_if<arma::is_arma_type<T>>::type*)
{
  return input + ".csv";
}

/**
 * Get the parameter name for a serializable model type (where the user has to
 * pass the file that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamValue(
    const util::ParamData& /* data */,
    const std::string& input,
    const typename boost::enable_if<data::HasSerialize<T>>::type*)
{
  return input + ".bin";
}

/**
 * Get the parameter name for a mapped matrix type (where the user has to pass
 * the file that holds the matrix).
 */
template<typename T>
std::string GetPrintableParamValue(
    const util::ParamData& /* data */,
    const std::string& input,
    const typename boost::enable_if<std::is_same<T,
        std::tuple<data::DatasetInfo, arma::mat>>>::type*)
{
  return input + ".arff";
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
