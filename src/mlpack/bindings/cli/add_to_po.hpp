/**
 * @file add_to_po.hpp
 * @author Ryan Curtin
 *
 * Utility functions to add options to CLI11 based on their
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CMD_ADD_TO_PO_HPP
#define MLPACK_BINDINGS_CMD_ADD_TO_PO_HPP

#include <mlpack/core/util/param_data.hpp>
#include <mlpack/core/util/is_std_vector.hpp>
#include "map_parameter_name.hpp"

#include <CLI/CLI.hpp>

namespace mlpack {
namespace bindings {
namespace cmd {

/**
 * Add a non-vector option to CLI11.
 *
 * @param boostName The name of the option to add to CLI11.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& cliName,
             const std::string& descr,
             CLI::App& app,
             const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
             const typename boost::disable_if<std::is_same<T, bool>>::type* = 0)
{
  T value;
  app.add_option(cliName.c_str(), value, descr.c_str());
}

/**
 * Add a vector option to CLI11.  This overload will use the
 * multitoken() option.
 *
 * @param boostName The name of the option to add to CLI11.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& cliName,
             const std::string& descr,
             CLI::App& app,
             const typename boost::enable_if<util::IsStdVector<T>>::type* = 0,
             const typename boost::disable_if<std::is_same<T, bool>>::type* = 0)
{
  T container;
  app.add_option(cliName.c_str(), container, descr.c_str());
}

/**
 * Add a boolean option to CLI11.
 *
 * @param boostName The name of the option to add to CLI11.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& cliName,
             const std::string& descr,
             CLI::App& app,
             const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
             const typename boost::enable_if<std::is_same<T, bool>>::type* = 0)
{
  bool flag; // Seems to be the prefered way,
  // Template substitution fais without it, why??
  app.add_flag(cliName.c_str(), flag, descr.c_str());
}

/**
 * Add an option to CLI11.  This is the function meant to be
 * used in the CMD function map.
 *
 * @param d Parameter data.
 * @param input Unused void pointer.
 * @param output Void pointer to options_description object.
 */
template<typename T>
void AddToPO(const util::ParamData& d,
             const void* /* input */,
             void* output)
{
  // Cast CMD::App object.
  CLI::App* app = (CLI::App*) output;

  // Generate the name to be given to CLI11.
  const std::string mappedName =
      MapParameterName<typename std::remove_pointer<T>::type>(d.name);
  std::string boostName = (d.alias != '\0') ? mappedName + "," +
      std::string(1, d.alias) : mappedName;

  // Note that we have to add the option as type equal to the mapped type, not
  // the true type of the option.
  AddToPO<typename ParameterType<typename std::remove_pointer<T>::type>::type>(
      boostName, d.desc, *app);
}

} // namespace cmd
} // namespace bindings
} // namespace mlpack

#endif
