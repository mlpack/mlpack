/**
 * @file bindings/cli/print_doc_functions_impl.hpp
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
 * Given the name of a binding, print its command-line name (this returns
 * "mlpack_<bindingName>".
 */
inline std::string GetBindingName(const std::string& bindingName)
{
  return "mlpack_" + bindingName;
}

/**
 * Print any imports for CLI (there are none, so this returns an empty string).
 */
inline std::string PrintImport(const std::string& /* bindingName */)
{
  return "";
}

/**
 * Print any special information about input options.
 */
inline std::string PrintInputOptionInfo()
{
  return "";
}
/**
 * Print any special information about output options.
 */
inline std::string PrintOutputOptionInfo()
{
  return "";
}

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
 * Given a vector parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const std::vector<T>& value, bool quotes)
{
  std::ostringstream oss;
  if (quotes)
    oss << "'";
  if (value.size() > 0)
  {
    oss << value[0];
    for (size_t i = 1; i < value.size(); ++i)
      oss << ", " << value[i];
  }
  if (quotes)
    oss << "'";
  return oss.str();
}

/**
 * Given a parameter name, print its corresponding default value.
 */
inline std::string PrintDefault(const std::string& paramName)
{
  if (IO::Parameters().count(paramName) == 0)
    throw std::invalid_argument("unknown parameter " + paramName + "!");

  util::ParamData& d = IO::Parameters()[paramName];

  std::string defaultValue;
  IO::GetSingleton().functionMap[d.tname]["DefaultParam"](d, NULL,
      (void*) &defaultValue);

  return defaultValue;
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
  if (IO::Parameters().count(paramName) > 0)
  {
    util::ParamData& d = IO::Parameters()[paramName];

    std::string name;
    IO::GetSingleton().functionMap[d.tname]["GetPrintableParamName"](d, NULL,
        (void*) &name);

    std::ostringstream ossValue;
    ossValue << value;
    std::string rawValue = ossValue.str();
    std::string fullValue;
    IO::GetSingleton().functionMap[d.tname]["GetPrintableParamValue"](d,
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
        "encountered while assembling documentation!  Check BINDING_LONG_DESC()"
        + " and BINDING_EXAMPLE() declaration.");
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
  return util::HyphenateString("$ " + GetBindingName(programName) + " " +
      ProcessOptions(args...), 2);
}

/**
 * Given a program name, print a program call invocation assuming that all
 * options are specified.
 */
inline std::string ProgramCall(const std::string& programName)
{
  std::ostringstream oss;
  oss << "$ " << GetBindingName(programName);

  // Handle all options---first input options, then output options.
  std::map<std::string, util::ParamData>& parameters = IO::Parameters();

  for (auto& it : parameters)
  {
    if (!it.second.input || it.second.persistent)
      continue;

    // Otherwise, print the name and the default value.
    std::string name;
    IO::GetSingleton().functionMap[it.second.tname]["GetPrintableParamName"](
        it.second, NULL, (void*) &name);

    std::string value;
    IO::GetSingleton().functionMap[it.second.tname]["DefaultParam"](
        it.second, NULL, (void*) &value);
    if (value == "''")
      value = "<string>";

    oss << " ";
    if (!it.second.required)
      oss << "[";

    oss << name;
    if (it.second.cppType != "bool")
      oss << " " << value;

    if (!it.second.required)
      oss << "]";
  }

  // Now get the output options.
  for (auto& it : parameters)
  {
    if (it.second.input)
      continue;

    // Otherwise, print the name and the default value.
    std::string name;
    IO::GetSingleton().functionMap[it.second.tname]["GetPrintableParamName"](
        it.second, NULL, (void*) &name);

    std::string value;
    IO::GetSingleton().functionMap[it.second.tname]["DefaultParam"](
        it.second, NULL, (void*) &value);
    if (value == "''")
      value = "<string>";

    oss << " [" << name;
    if (it.second.cppType != "bool")
      oss << " " << value;
    oss << "]";
  }

  return util::HyphenateString(oss.str(), 8);
}

/**
 * Print what a user would type to invoke the given option name.  Note that the
 * name *must* exist in the CLI module.  (Note that because of the way
 * BINDING_LONG_DESC() and BINDING_EXAMPLE() is structured, this doesn't mean
 * that all of the PARAM_*() declarataions need to come before
 * BINDING_LONG_DESC() and BINDING_EXAMPLE() declaration.)
 */
inline std::string ParamString(const std::string& paramName)
{
  // Return the correct parameter name.
  if (IO::Parameters().count(paramName) > 0)
  {
    util::ParamData& d = IO::Parameters()[paramName];

    std::string output;
    IO::GetSingleton().functionMap[d.tname]["GetPrintableParamName"](d, NULL,
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
        "BINDING_LONG_DESC() and BINDING_EXAMPLE() definition.");
  }
}

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
