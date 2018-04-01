/**
 * @file print_doc_functions_impl.hpp
 * @author Ryan Curtin
 *
 * This will generate a string representing what a user should type to invoke a
 * given option.  For the command-line bindings, this will generate strings like
 * '--param_name=x' or '--param_name'.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes)
{
  std::ostringstream oss;
  if (quotes)
    oss << "'";
  oss << value;
  if (quotes)
    oss << "'";
  return oss.str();
}

/**
 * Print a dataset type parameter (add .csv and return).
 */
inline std::string PrintDataset(const std::string& dataset)
{
  return "'" + dataset + ".csv'";
}

/**
 * Print a model type parameter (add .bin and return).
 */
inline std::string PrintModel(const std::string& model)
{
  return "'" + model + ".bin'";
}

// Base case for recursion.
inline std::string ProcessOptions() { return ""; }

/**
 * Print an option for a command-line argument.
 */
template<typename T, typename... Args>
std::string ProcessOptions(const std::string& paramName,
                           const T& value,
                           Args... args)
{
  // See if it is part of the program.
  std::string result = "";
  if (CLI::Parameters().count(paramName) > 0)
  {
    const util::ParamData& d = CLI::Parameters()[paramName];

    std::string name;
    CLI::GetSingleton().functionMap[d.tname]["GetPrintableParamName"](d, NULL,
        (void*) &name);

    std::ostringstream ossValue;
    ossValue << value;
    std::string rawValue = ossValue.str();
    std::string fullValue;
    CLI::GetSingleton().functionMap[d.tname]["GetPrintableParamValue"](d,
        (void*) &rawValue, (void*) &fullValue);

    std::ostringstream oss;
    if (d.tname != TYPENAME(bool))
      oss << name << " " << fullValue;
    else
      oss << name;
    result = oss.str();
  }
  else
  {
    throw std::runtime_error("Unknown parameter '" + paramName + "' " +
        "encountered while assembling documentation!  Check PROGRAM_INFO() " +
        "declaration.");
  }

  std::string rest = ProcessOptions(args...);
  if (rest != "")
    result += " " + rest;

  return result;
}

/**
 * Given a program name and arguments for it, print what its invocation would
 * be.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args)
{
  return util::HyphenateString("$ " + programName + " " +
      ProcessOptions(args...), 2);
}

/**
 * Print what a user would type to invoke the given option name.  Note that the
 * name *must* exist in the CLI module.  (Note that because of the way
 * ProgramInfo is structured, this doesn't mean that all of the PARAM_*()
 * declarataions need to come before the PROGRAM_INFO() declaration.)
 */
inline std::string ParamString(const std::string& paramName)
{
  // Return the correct parameter name.
  if (CLI::Parameters().count(paramName) > 0)
  {
    util::ParamData& d = CLI::Parameters()[paramName];

    std::string output;
    CLI::GetSingleton().functionMap[d.tname]["GetPrintableParamName"](d, NULL,
        (void*) &output);
    // Is there an alias?
    std::string alias = "";
    if (d.alias != '\0')
      alias = " (-" + std::string(1, d.alias) + ")";

    return "'" + output + alias + "'";
  }
  else
  {
    throw std::runtime_error("Parameter '" + paramName + "' not known!  Check "
        "PROGRAM_INFO() definition.");
  }
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
