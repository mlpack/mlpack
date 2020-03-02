/**
 * @file print_doc_functions_impl.hpp
 * @author Yashwant Singh
 * @author Yasmine Dumouchel
 *
 * This file contains functions useful for printing documentation strings
 * related to Go bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_GO_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_GO_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include <mlpack/core/util/hyphenate_string.hpp>
#include "strip_type.hpp"
#include "camel_case.hpp"

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Given the name of a binding, print its Go name.
 */
inline std::string GetBindingName(const std::string& bindingName)
{
  // No modification is needed to the name---we just use it as-is.
  return CamelCase(bindingName) + "()";
}

/**
 * Print any import information for the Go binding.
 */
inline std::string PrintImport()
{
  return "import (\n"
      "  \"mlpack/build/src/mlpack/bindings/go/mlpack\"\n"
      "  \"gonum.org/v1/gonum/mat\"\n"
      ")";
}

/**
 * Print any special information about output options.
 */
inline std::string PrintOutputOptionInfo()
{
  return "Output options are returned via Go's support for multiple "
         "return values.";
}

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes)
{
  std::ostringstream oss;
  if (quotes)
    oss << "\"";
  oss << value;
  if (quotes)
    oss << "\"";
  return oss.str();
}

// Special overload for booleans.
template<>
inline std::string PrintValue(const bool& value, bool quotes)
{
  if (quotes && value)
    return "\"true\"";
  else if (quotes && !value)
    return "\"false\"";
  else if (!quotes && value)
    return "true";
  else
    return "false";
}

/**
 * Given a parameter name, print its corresponding default value.
 */
inline std::string PrintDefault(const std::string& paramName)
{
  if (CLI::Parameters().count(paramName) == 0)
    throw std::invalid_argument("unknown parameter " + paramName + "!");

  const util::ParamData& d = CLI::Parameters()[paramName];

  std::string defaultValue;
  CLI::GetSingleton().functionMap[d.tname]["DefaultParam"](d, NULL,
      (void*) &defaultValue);

  return defaultValue;
}

// Recursion base case.
std::string PrintOptionalInputs() { return ""; }

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in CLI.
 */
template<typename T, typename... Args>
std::string PrintOptionalInputs(const std::string& paramName,
                              const T& value,
                              Args... args)
{
  // See if this is part of the program.
  std::string result = "";
  if (CLI::Parameters().count(paramName) > 0)
  {
    const util::ParamData& d = CLI::Parameters()[paramName];
    if (d.input && !d.required)
    {
      std::string goParamName = CamelCase(paramName);

      // Print the input option.
      std::ostringstream oss;
      oss << "  param." << goParamName << " = ";

      // Special handling is needed for model types.
      std::string name;
      CLI::GetSingleton().functionMap[d.tname]["GetType"](d, NULL,
         (void*) &name);
      if (name[name.size() - 1] == '*')
      {
        oss << "&";
        oss << CamelCase(PrintValue(value, d.tname == TYPENAME(std::string)));
      }
      else
      {
        oss << PrintValue(value, d.tname == TYPENAME(std::string));
      }
      oss << "\n";
      result = oss.str();
    }
  }
  else
  {
    // Unknown parameter!
    throw std::runtime_error("Unknown parameter '" + paramName + "' " +
        "encountered while assembling documentation!  Check PROGRAM_INFO() " +
        "declaration.");
  }

  // Continue recursion.
  std::string rest = PrintOptionalInputs(args...);
  if (rest != "" && result != "")
    result += rest;
  else if (result == "")
    result = rest;

  return result;
}

// Recursion base case.
std::string PrintInputOptions() { return ""; }

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in CLI.
 */
template<typename T, typename... Args>
std::string PrintInputOptions(const std::string& paramName,
                              const T& value,
                              Args... args)
{
  // See if this is part of the program.
  std::string result = "";

  if (CLI::Parameters().count(paramName) > 0)
  {
    const util::ParamData& d = CLI::Parameters()[paramName];
    if (d.input && d.required)
    {
      // Print the input option.
      std::ostringstream oss;
      std::string name;
      CLI::GetSingleton().functionMap[d.tname]["GetType"](d, NULL,
         (void*) &name);
      if (name[name.size() - 1] == '*')
      {
        oss << "&";
        oss << CamelCase(PrintValue(value, d.tname == TYPENAME(std::string)));
      }
      else
      {
        oss << PrintValue(value, d.tname == TYPENAME(std::string));
      }
      result = oss.str();
    }
  }
  else
  {
    // Unknown parameter!
    throw std::runtime_error("Unknown parameter '" + paramName + "' " +
        "encountered while assembling documentation!  Check PROGRAM_INFO() " +
        "declaration.");
  }

  // Continue recursion.
  std::string rest = PrintInputOptions(args...);
  if (rest != "" && result != "")
    result += ", " + rest;
  else if (result == "")
    result = rest;

  return result;
}

// Base case: no modification needed.
void GetOptions(
    std::vector<std::tuple<std::string, std::string>>& /* results */)
{
  // Nothing to do.
}

/**
 * Assemble a vector of string tuples indicating parameter names and what should
 * be printed for them.  (For output parameters, we just need to print the
 * value.)
 */
template<typename T, typename... Args>
void GetOptions(
    std::vector<std::tuple<std::string, std::string>>& results,
    const std::string& paramName,
    const T& value,
    Args... args)
{
  // Determine whether or not the value is required.
  if (CLI::Parameters().count(paramName) > 0)
  {
    std::ostringstream oss;
    oss << value;
    results.push_back(std::make_tuple(paramName, oss.str()));
    GetOptions(results, args...);
  }
  else
  {
    // Unknown parameter!
    throw std::runtime_error("Unknown parameter '" + paramName + "' " +
        "encountered while assembling documentation!  Check PROGRAM_INFO() " +
        "declaration.");
  }
}

// Recursion base case.
inline std::string PrintOutputOptions() { return ""; }

template<typename... Args>
std::string PrintOutputOptions(Args... args)
{
  // Get the list of output options for the binding.
  std::vector<std::string> outputOptions;
  for (auto it = CLI::Parameters().begin(); it != CLI::Parameters().end(); ++it)
  {
    const util::ParamData& d = it->second;
    if (!d.input)
      outputOptions.push_back(it->first);
  }

  // Now get the full list of output options that we have.
  std::vector<std::tuple<std::string, std::string>> passedOptions;
  GetOptions(passedOptions, args...);

  // Next, iterate over all the options.
  std::ostringstream oss;
  for (size_t i = 0; i < outputOptions.size(); ++i)
  {
    // Does this option exist?
    bool found = false;
    size_t index = passedOptions.size();
    for (size_t j = 0; j < passedOptions.size(); ++j)
    {
      if (outputOptions[i] == std::get<0>(passedOptions[j]))
      {
        found = true;
        index = j;
        break;
      }
    }

    if (found)
    {
      // We have received this option, so print it.
      if (i == 0)
      {
        oss << "  " << CamelCase(std::get<1>(passedOptions[index]));
      }
      else if (i > 0)
      {
        oss << ", ";
        oss << CamelCase(std::get<1>(passedOptions[index]));
      }
    }
    else
    {
      // We don't care about this option.
      if (i == 0)
      {
        oss << "  _";
      }
      else if (i > 0)
      {
        oss << ", _";
      }
    }
  }

  return oss.str();
}

/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args)
{
  std::string result = "";
  std::string goProgramName = CamelCase(programName);

  // Initialize the method parameter structure
  std::ostringstream oss;
  oss << "  param := mlpack.Initialize" << goProgramName << "()\n";
  result = oss.str();
  oss.str(""); // Reset it.

  // Now process each optional parameters.
  oss << PrintOptionalInputs(args...);
  std::string param = oss.str();
  result = result + util::HyphenateString(param, 0);
  oss.str(""); // Reset it.

  // Now process each output parameters.
  oss << PrintOutputOptions(args...);
  std::string output = oss.str();
  result = result + util::HyphenateString(output, 0);
  oss.str(""); // Reset it.

  oss << " := mlpack." << goProgramName << "(";
  result = result + oss.str();
  oss.str(""); // Reset it.

  // Now process each required input parameter.
  oss << PrintInputOptions(args...);
  std::string input = oss.str();
  if (input != "")
    result = result + input + ", ";
  oss.str(""); // Reset it.
  if (param != "")
    result = result + "param";

  result = result + ")";
  return result;
}

/**
 * Given the name of a model, print it.  Here we do not need to modify anything.
 */
inline std::string PrintModel(const std::string& modelName)
{
  std::string goStrippedType, strippedType, printedType, defaultsType;
  StripType(modelName, goStrippedType, strippedType, printedType, defaultsType);

  return strippedType;
}

/**
 * Given the name of a matrix, print it.  Here we do not need to modify
 * anything.
 */
inline std::string PrintDataset(const std::string& datasetName)
{
  return datasetName;
}

/**
 * Given the name of a binding, print its invocation.
 */
inline std::string ProgramCall(const std::string& programName)
{
  std::ostringstream oss;
  std::string goProgramName = CamelCase(programName);

  // Determine if we have any output options.
  const std::map<std::string, util::ParamData>& parameters = CLI::Parameters();

  oss << "  param := mlpack.Initialize" << goProgramName << "()\n";

  std::vector<std::string> outputOptions;
  for (auto it = CLI::Parameters().begin(); it != CLI::Parameters().end(); ++it)
  {
    const util::ParamData& d = it->second;
    if (!d.input)
      outputOptions.push_back(it->first);
  }
  std::string result = oss.str();
  oss.str("");
  std::string param = "";
  // Now iterate over every input option.
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    if (it->second.input && !it->second.required && !it->second.persistent)
    {
      // Print the input option.
      oss << "  param." << CamelCase(it->second.name) << " = ";
      std::string value;
      CLI::GetSingleton().functionMap[it->second.tname]["DefaultParam"](
          it->second, NULL, (void*) &value);
      oss << value;
      oss << "\n";
      param = oss.str();
    }
  }

  // Now iterate over every optional input option.
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    if (it->second.input && it->second.required && !it->second.persistent)
    {
      // Print the input option.
      oss << "  " << CamelCase(it->second.name) << " := ";
      std::string value;
      CLI::GetSingleton().functionMap[it->second.tname]["DefaultParam"](
          it->second, NULL, (void*) &value);
      oss << value;
      oss << "\n";
    }
  }

  result += oss.str();
  oss.str("");
  oss << result;

  // Now print output lines.
  size_t outputs = 0;
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    if (!it->second.input)
    {
      if (outputs > 0)
      {
        oss << ", ";
        oss << CamelCase(it->second.name);
      }
      else
      {
        oss << "  " << CamelCase(it->second.name);
      }
      ++outputs;
    }
  }

  oss << " := mlpack." << goProgramName << "(";
  for (auto i = parameters.begin(); i != parameters.end(); ++i)
  {
    if (i->second.input && i->second.required && i != parameters.end())
      oss << CamelCase(i->second.name) << ", ";
    else if (i == parameters.end())
      oss << CamelCase(i->second.name);
  }
  if (param != "")
    oss << "param";
  oss << ")\n";

  result = "";
  result = util::HyphenateString(oss.str(), 0);

  oss.str("");
  oss << result;

  return oss.str();
}

/**
 * Print any closing call to a program.
 */
inline std::string ProgramCallClose()
{
  return ")";
}

/**
 * Given the parameter name, determine what it would actually be when passed to
 * the command line.
 */
inline std::string ParamString(const std::string& paramName)
{
  // For a Go binding we don't need to know the type.
  return "\"" + paramName + "\"";
}

/**
 * Given the parameter name and an argument, return what should be written as
 * documentation when referencing that argument.
 */
template<typename T>
inline std::string ParamString(const std::string& paramName, const T& value)
{
  const util::ParamData& d = CLI::Parameters()[paramName];
  std::ostringstream oss;
  oss << paramName << "="
      << PrintValue(value, d.tname == TYPENAME(std::string));
  return oss.str();
}

inline bool IgnoreCheck(const std::string& paramName)
{
  return !CLI::Parameters()[paramName].input;
}

inline bool IgnoreCheck(const std::vector<std::string>& constraints)
{
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!CLI::Parameters()[constraints[i]].input)
      return true;
  }

  return false;
}

inline bool IgnoreCheck(
    const std::vector<std::pair<std::string, bool>>& constraints,
    const std::string& paramName)
{
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!CLI::Parameters()[constraints[i].first].input)
      return true;
  }

  return !CLI::Parameters()[paramName].input;
}

} // namespace go
} // namespace bindings
} // namespace mlpack

#endif
