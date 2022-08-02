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
#ifndef MLPACK_BINDINGS_CLI_ADD_TO_CLI11_HPP
#define MLPACK_BINDINGS_CLI_ADD_TO_CLI11_HPP

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
 * @param param an object of util::ParamData.
 * @param app A CLI11 object to add parameter to.
 */
template<typename T>
void AddToCLI11(const std::string& cliName,
                util::ParamData& param,
                CLI::App& app,
                const typename std::enable_if<!std::is_same<T,
                    bool>::value>::type* = 0,
                const typename std::enable_if<!
                    arma::is_arma_type<T>::value>::type* = 0,
                const typename std::enable_if<!
                    data::HasSerialize<T>::value>::type* = 0,
                const typename std::enable_if<std::is_same<T,
                    std::tuple<mlpack::data::DatasetInfo,
                    arma::mat>>::value>::type* = 0)
{
  app.add_option_function<std::string>(cliName.c_str(),
      [&param](const std::string& value)
      {
        using TupleType = std::tuple<T, typename ParameterType<T>::type>;
        TupleType& tuple = *MLPACK_ANY_CAST<TupleType>(&param.value);
        std::get<0>(std::get<1>(tuple)) = MLPACK_ANY_CAST<std::string>(value);
        param.wasPassed = true;
      },
      param.desc.c_str());
}

/**
 * Add a serializable option to CLI11.
 *
 * @param cliName The name of the option to add to CLI11.
 * @param param an object of util::ParamData.
 * @param app a CLI11 object to add parameter to.
 */
template<typename T>
void AddToCLI11(const std::string& cliName,
                util::ParamData& param,
                CLI::App& app,
                const typename std::enable_if<!std::is_same<T,
                    bool>::value>::type* = 0,
                const typename std::enable_if<!
                    arma::is_arma_type<T>::value>::type* = 0,
                const typename std::enable_if<
                    data::HasSerialize<T>::value>::type* = 0,
                const typename std::enable_if<!std::is_same<T,
                    std::tuple<mlpack::data::DatasetInfo,
                    arma::mat>>::value>::type* = 0)
{
  app.add_option_function<std::string>(cliName.c_str(),
      [&param](const std::string& value)
      {
        using TupleType = std::tuple<T*, typename ParameterType<T>::type>;
        TupleType& tuple = *MLPACK_ANY_CAST<TupleType>(&param.value);
        std::get<1>(tuple) = MLPACK_ANY_CAST<std::string>(value);
        param.wasPassed = true;
      },
      param.desc.c_str());
}

/**
 * Add an arma matrix to CLI11.
 *
 * @param cliName The name of the option to add to CLI11.
 * @param param an object of util::ParamData.
 * @param app a CLI11 object to add parameter to.
 */
template<typename T>
void AddToCLI11(const std::string& cliName,
                util::ParamData& param,
                CLI::App& app,
                const typename std::enable_if<!
                    std::is_same<T, bool>::value>::type* = 0,
                const typename std::enable_if<
                    arma::is_arma_type<T>::value>::type* = 0,
                const typename std::enable_if<!std::is_same<T,
                  std::tuple<mlpack::data::DatasetInfo,
                    arma::mat>>::value>::type* = 0)
{
  app.add_option_function<std::string>(cliName.c_str(),
      [&param](const std::string& value)
      {
        using TupleType = std::tuple<T, typename ParameterType<T>::type>;
        TupleType& tuple = *MLPACK_ANY_CAST<TupleType>(&param.value);
        std::get<0>(std::get<1>(tuple)) = MLPACK_ANY_CAST<std::string>(value);
        param.wasPassed = true;
      },
      param.desc.c_str());
}

/**
 * Add an option to CLI11.
 *
 * @param cliName The name of the option to add to CLI11.
 * @param param an object of util::ParamData.
 * @param app a CLI11 object to add parameter to.
 */
template<typename T>
void AddToCLI11(const std::string& cliName,
                util::ParamData& param,
                CLI::App& app,
                const typename std::enable_if<!
                    std::is_same<T, bool>::value>::type* = 0,
                const typename std::enable_if<!
                    arma::is_arma_type<T>::value>::type* = 0,
                const typename std::enable_if<!
                    data::HasSerialize<T>::value>::type* = 0,
                const typename std::enable_if<!std::is_same<T,
                    std::tuple<mlpack::data::DatasetInfo,
                    arma::mat>>::value>::type* = 0)
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
 * @param param an object of util::ParamData.
 * @param app a CLI11 object to add parameter to.
 */
template<typename T>
void AddToCLI11(const std::string& cliName,
                util::ParamData& param,
                CLI::App& app,
                const typename std::enable_if<
                    std::is_same<T, bool>::value>::type* = 0,
                const typename std::enable_if<!
                    arma::is_arma_type<T>::value>::type* = 0,
                const typename std::enable_if<!
                    data::HasSerialize<T>::value>::type* = 0,
                const typename std::enable_if<!std::is_same<T,
                    std::tuple<mlpack::data::DatasetInfo,
                    arma::mat>>::value>::type* = 0)
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
 * @param param Parameter data.
 * @param * (input) Unused void pointer.
 * @param output Void pointer to options_description object.
 */
template<typename T>
void AddToCLI11(util::ParamData& param,
                const void* /* input */,
                void* output)
{
  // Cast CLI::App object.
  CLI::App* app = (CLI::App*) output;

  // Generate the name to be given to CLI11.
  const std::string mappedName =
      MapParameterName<typename std::remove_pointer<T>::type>(param.name);
  std::string cliName = (param.alias != '\0') ?
      "-" + std::string(1, param.alias) + ",--" + mappedName :
      "--" + mappedName;

  // Note that we have to add the option as type equal to the mapped type, not
  // the true type of the option.
  AddToCLI11<typename std::remove_pointer<T>::type>(
      cliName, param, *app);
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
