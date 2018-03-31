/**
 * @file add_to_po.hpp
 * @author Ryan Curtin
 *
 * Utility functions to add options to boost::program_options based on their
 * type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_ADD_TO_PO_HPP
#define MLPACK_BINDINGS_CLI_ADD_TO_PO_HPP

#include <mlpack/core/util/param_data.hpp>
#include <boost/program_options.hpp>
#include <mlpack/core/util/is_std_vector.hpp>
#include "map_parameter_name.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Add a non-vector option to boost::program_options.
 *
 * @param boostName The name of the option to add to boost::program_options.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& boostName,
             const std::string& descr,
             boost::program_options::options_description& desc,
             const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
             const typename boost::disable_if<std::is_same<T, bool>>::type* = 0)
{
  desc.add_options()(boostName.c_str(), boost::program_options::value<T>(),
      descr.c_str());
}

/**
 * Add a vector option to boost::program_options.  This overload will use the
 * multitoken() option.
 *
 * @param boostName The name of the option to add to boost::program_options.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& boostName,
             const std::string& descr,
             boost::program_options::options_description& desc,
             const typename boost::enable_if<util::IsStdVector<T>>::type* = 0,
             const typename boost::disable_if<std::is_same<T, bool>>::type* = 0)
{
  desc.add_options()(boostName.c_str(),
      boost::program_options::value<T>()->multitoken(), descr.c_str());
}

/**
 * Add a boolean option to boost::program_options.
 *
 * @param boostName The name of the option to add to boost::program_options.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& boostName,
             const std::string& descr,
             boost::program_options::options_description& desc,
             const typename boost::disable_if<util::IsStdVector<T>>::type* = 0,
             const typename boost::enable_if<std::is_same<T, bool>>::type* = 0)
{
  desc.add_options()(boostName.c_str(), descr.c_str());
}

/**
 * Add an option to boost::program_options.  This is the function meant to be
 * used in the CLI function map.
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
  // Cast boost::program_options::options_description object.
  boost::program_options::options_description* desc =
      (boost::program_options::options_description*) output;

  // Generate the name to be given to boost::program_options.
  const std::string mappedName =
      MapParameterName<typename std::remove_pointer<T>::type>(d.name);
  std::string boostName = (d.alias != '\0') ? mappedName + "," +
      std::string(1, d.alias) : mappedName;

  // Note that we have to add the option as type equal to the mapped type, not
  // the true type of the option.
  AddToPO<typename ParameterType<typename std::remove_pointer<T>::type>::type>(
      boostName, d.desc, *desc);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
