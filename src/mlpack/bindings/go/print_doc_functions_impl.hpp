/**
 * @file print_doc_functions_impl.hpp
 * @author Ryan Curtin
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

namespace mlpack {
namespace bindings {
namespace go {

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
      std::string goParamName = paramName;
      goParamName[0] = std::toupper(goParamName[0]);

      // Print the input option.
      std::ostringstream oss;
      oss << ">>> " << "param." << goParamName << " = ";
      oss << PrintValue(value, d.tname == TYPENAME(std::string));
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
      oss << PrintValue(value, d.tname == TYPENAME(std::string));
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

// Recursion base case.
inline std::string PrintOutputOptions() { return ""; }

template<typename T, typename... Args>
std::string PrintOutputOptions(const std::string& paramName,
                               const T& value,
                               Args... args)
{
  // See if this is part of the program.
  std::string result = "";
  if (CLI::Parameters().count(paramName) > 0)
  {
    const util::ParamData& d = CLI::Parameters()[paramName];
    if (!d.input)
    {
      // Print a new line for the output option.
      std::ostringstream oss;
      oss << value;
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
  std::string rest = PrintOutputOptions(args...);
  if (rest != "" && result != "")
    result += ", " + rest;
  else if (result == "")
    result = rest;

  return result;
}


/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args)
{
  std::string result = "";
  std::string goProgramName = programName;
  goProgramName[0] = std::toupper(goProgramName[0]);

  // Initialize the method parameter structure
  std::ostringstream oss;
  oss << ">>> param := Initialize" << goProgramName << "()\n";
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
  result = result + ">>> " + util::HyphenateString(output, 0);
  oss.str(""); // Reset it.

  oss << " := " << goProgramName << "(";
  result = result + oss.str();
  oss.str(""); // Reset it.

  // Now process each input required parameters.
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
  std::string goModelName = modelName;
  const size_t loc = goModelName.find("<>");
  if (loc != std::string::npos)
  {
    // Convert it from "<>".
    goModelName.replace(loc, 2, "");
  }
  return goModelName;
}

/**
 * Given the name of a matrix, print it.  Here we do not need to modify
 * anything.
 */
inline std::string PrintDataset(const std::string& datasetName)
{
  return "" + datasetName + "";
}

/**
 * Given the name of a binding, print its invocation.
 */
inline std::string ProgramCall(const std::string& programName)
{
  std::string goProgramName = programName;
  goProgramName[0] = std::toupper(goProgramName[0]);
  return goProgramName + "(";
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

  return "'" + paramName + "'";
}

/**
 * Given the parameter name and an argument, return what should be written as
 * documentation when referencing that argument.
 */
template<typename T>
inline std::string ParamString(const std::string& paramName, const T& value)
{
  std::ostringstream oss;
    oss << paramName << "=" << value;
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
