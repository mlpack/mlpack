/**
 * @file bindings/cli/add_to_cli11.hpp
 * @author Ryan Curtin
 *
 * Utility functions to add options to CLI11 based on their type.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_ADD_TO_PO_HPP
#define MLPACK_BINDINGS_CLI_ADD_TO_PO_HPP

#include <mlpack/core/util/param_data.hpp>
#include <mlpack/core/util/is_std_vector.hpp>
#include "map_parameter_name.hpp"

#include "third_party/CLI/CLI11.hpp"

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Add a tuple option to CLI11.
 * 
 * @param cliName The name of the option to add to CLI11.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& cliName,
             util::ParamData& param,
             CLI::App& app,
             const typename boost::disable_if<std::is_same<T, bool>>::type* = 0,
             const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
             const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
             const typename boost::enable_if<std::is_same<T,
                std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
{
  app.add_option_function<T>(cliName.c_str(),
      [&param](const T& value)
      {
        typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
        TupleType& tuple = *boost::any_cast<TupleType>(&param.value);
        std::get<1>(tuple) = boost::any_cast<std::string>(value);
        param.wasPassed = true;
      },
      param.desc.c_str());
}

/**
 * Add a serializable option to CLI11.
 * 
 * @param cliName The name of the option to add to CLI11.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& cliName,
             util::ParamData& param,
             CLI::App& app,
             const typename boost::disable_if<std::is_same<T, bool>>::type* = 0,
             const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
             const typename boost::enable_if<data::HasSerialize<T>>::type* = 0,
             const typename boost::disable_if<std::is_same<T,
                std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
{
  app.add_option_function<T>(cliName.c_str(),
      [&param](const T& value)
      {
        using TupleType = std::tuple<T*, typename ParameterType<T>::type>;
        TupleType& tuple = boost::any_cast<TupleType>(&param.value);
        std::get<1>(tuple) = boost::any_cast<std::string>(value);
      },
      param.desc.c_str());
}

/**
 * Add an arma matrix to CLI11.
 * 
 * @param cliName The name of the option to add to CLI11.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& cliName,
             util::ParamData& param,
             CLI::App& app,
             const typename boost::disable_if<std::is_same<T, bool>>::type* = 0,
             const typename boost::enable_if<arma::is_arma_type<T>>::type* = 0,
             const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
             const typename boost::disable_if<std::is_same<T,
                std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
{
  app.add_option_function<T>(cliName.c_str(),
      [&param](const T& value)
      { 
        typedef std::tuple<T, typename ParameterType<T>::type> TupleType;
        TupleType& tuple = *boost::any_cast<TupleType>(&param.value);
        std::get<1>(tuple) = boost::any_cast<std::string>(value);
        param.wasPassed = true;
      },
      param.desc.c_str());
}

/**
 * Add an option to CLI11.
 * 
 * @param cliName The name of the option to add to CLI11.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& cliName,
             util::ParamData& param,
             CLI::App& app,
             const typename boost::disable_if<std::is_same<T, bool>>::type* = 0,
             const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
             const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
             const typename boost::disable_if<std::is_same<T,
                std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
{
  app.add_option_function<T>(cliName.c_str(),
      [&param](const T& value)
      {
        param.value = value;
        param.wasPassed = true;
      },
      param.desc.c_str());
}

/**
 * Add a boolean option to CLI11.
 *
 * @param cliName The name of the option to add to CLI11.
 * @param descr Description string for parameter.
 * @param desc Options description to add parameter to.
 */
template<typename T>
void AddToPO(const std::string& cliName,
             util::ParamData& param,
             CLI::App& app,
             const typename boost::enable_if<std::is_same<T, bool>>::type* = 0,
             const typename boost::disable_if<arma::is_arma_type<T>>::type* = 0,
             const typename boost::disable_if<data::HasSerialize<T>>::type* = 0,
             const typename boost::disable_if<std::is_same<T,
                std::tuple<mlpack::data::DatasetInfo, arma::mat>>>::type* = 0)
{
  app.add_flag_function(cliName.c_str(),
      [&param](const T& value)
      {
        param.value = value;
        param.wasPassed = true;
      },
      param.desc.c_str());
}

/**
 * Add an option to CLI11.  This is the function meant to be
 * used in the CLI function map.
 *
 * @param d Parameter data.
 * @param * (input) Unused void pointer.
 * @param output Void pointer to options_description object.
 */
template<typename T>
void AddToPO(util::ParamData& param,
             const void* /* input */,
             void* output)
{
  // Cast CLI::App object.
  CLI::App* app = (CLI::App*) output;

  // Generate the name to be given to CLI11.
  const std::string mappedName =
      MapParameterName<typename std::remove_pointer<T>::type>(param.name);
  std::string cliName = (param.alias != '\0') ?
      "-" + std::string(1, param.alias) + ",--" + mappedName : "--" + mappedName;

  // Note that we have to add the option as type equal to the mapped type, not
  // the true type of the option.  
  AddToPO<typename ParameterType<typename std::remove_pointer<T>::type>::type>(
      cliName, param, *app);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
