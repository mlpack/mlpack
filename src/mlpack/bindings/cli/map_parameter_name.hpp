/**
 * @file bindings/cli/map_parameter_name.hpp
 * @author Ryan Curtin
 *
 * Map a parameter name to what it seen by CLI11 using template
 * metaprogramming.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_MAP_PARAMETER_NAME_HPP
#define MLPACK_BINDINGS_CLI_MAP_PARAMETER_NAME_HPP

#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * If needed, map the parameter name to the name that is used by
 * CLI11.  This overload simply returns the same name, so it is
 * used for primitive types.
 */
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const std::enable_if_t<!arma::is_arma_type<T>::value>* = 0,
    const std::enable_if_t<!data::HasSerialize<T>::value>* = 0,
    const std::enable_if_t<!std::is_same_v<T,
        std::tuple<mlpack::data::DatasetInfo, arma::mat>>>* = 0)
{
  return identifier;
}

/**
 * Map the parameter name to the name that is used by CLI11.
 * This overload addresses matrices and models, where the parameter name has
 * "_file" appended to it (since a filename will be provided).
 */
template<typename T>
std::string MapParameterName(
    const std::string& identifier,
    const std::enable_if_t<
        arma::is_arma_type<T>::value ||
        std::is_same_v<T, std::tuple<mlpack::data::DatasetInfo, arma::mat>> ||
        data::HasSerialize<T>::value>* /* junk */ = 0)
{
  return identifier + "_file";
}

/**
 * Map the parameter name to the name seen by CLI.
 *
 * @param d Parameter data.
 * @param * (input) Unused parameter.
 * @param output Pointer to std::string that will hold the mapped name.
 */
template<typename T>
void MapParameterName(util::ParamData& d,
                      const void* /* input */,
                      void* output)
{
  // Store the mapped name in the output pointer, which is actually a string
  // pointer.
  *((std::string*) output) =
      MapParameterName<std::remove_pointer_t<T>>(d.name);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
