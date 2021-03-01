/**
 * @file bindings/R/print_doc_functions_impl.hpp
 * @author Yashwant Singh Parihar
 *
 * This file contains functions useful for printing documentation strings
 * related to R bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_R_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Given the name of a binding, print its R name.
 */
inline std::string GetBindingName(const std::string& bindingName)
{
  // No modification is needed to the name---we just use it as-is.
  return bindingName + "()";
}

/**
 * Print any import information for the R binding.
 */
inline std::string PrintImport()
{
  return "library(mlpack)";
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
  return "Results are returned in a R list.  The keys of the "
      "list are the names of the output parameters.";
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

/**
 * Given a vector parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const std::vector<T>& value, bool quotes)
{
  std::ostringstream oss;
  if (quotes)
    oss << "\"";
  oss << "c(";
  if (value.size() > 0)
  {
    oss << value[0];
    for (size_t i = 1; i < value.size(); ++i)
      oss << ", " << value[i];
  }
  oss << ")";
  if (quotes)
    oss << "\"";
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
 * Special overload for booleans.
 */
template<>
inline std::string PrintValue(const bool& value, bool quotes)
{
  if (quotes && value)
    return "\"TRUE\"";
  else if (quotes && !value)
    return "\"FALSE\"";
  else if (!quotes && value)
    return "TRUE";
  else
    return "FALSE";
}

/**
 * Recursion base case.
 */
std::string PrintInputOptions() { return ""; }

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in IO.
 */
template<typename T, typename... Args>
std::string PrintInputOptions(const std::string& paramName,
                              const T& value,
                              Args... args)
{
  // See if this is part of the program.
  std::string result = "";
  if (IO::Parameters().count(paramName) > 0)
  {
    util::ParamData& d = IO::Parameters()[paramName];
    if (d.input)
    {
      // Print the input option.
      std::ostringstream oss;
      oss << paramName << "=";
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

/**
 * Recursion base case.
 */
inline std::string PrintOutputOptions(const bool /* markdown */) { return ""; }

template<typename T, typename... Args>
std::string PrintOutputOptions(const bool markdown,
                               const std::string& paramName,
                               const T& value,
                               Args... args)
{
  // See if this is part of the program.
  std::string result = "";
  std::string command_prefix = "R> ";
  if (IO::Parameters().count(paramName) > 0)
  {
    util::ParamData& d = IO::Parameters()[paramName];
    if (!d.input)
    {
      // Print a new line for the output option.
      std::ostringstream oss;
      if (markdown)
        oss << command_prefix;
      oss << value << " <- output$" << paramName;
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
  std::string rest = PrintOutputOptions(markdown, args...);
  if (rest != "" && result != "")
    result += "\n";
  result += rest;

  return result;
}


/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
std::string ProgramCall(const bool markdown,
                        const std::string& programName,
                        Args... args)
{
  std::ostringstream oss;
  if (markdown)
    oss << "R> ";

  // Find out if we have any output options first.
  std::ostringstream ossOutput;
  ossOutput << PrintOutputOptions(markdown, args...);
  if (ossOutput.str() != "")
    oss << "output <- ";
  oss << programName << "(";

  // Now process each input option.
  oss << PrintInputOptions(args...);
  oss << ")";

  std::string call = oss.str();
  oss.str(""); // Reset it.

  // Now process each output option.
  oss << PrintOutputOptions(markdown, args...);
  if (markdown)
  {
    if (oss.str() == "")
      return util::HyphenateString(call, 2);
    else
      return util::HyphenateString(call, 2) + "\n" + oss.str();
  }

  if (oss.str() == "")
    return "\\dontrun{\n" + util::HyphenateString(call, 2) + "\n}";
  else
    return "\\dontrun{\n" + util::HyphenateString(call, 2) + "\n" + oss.str() +
           "\n}";
}

/**
 * Given the name of a binding, print a program call assuming that all options
 * are specified.  The programName should not be the output of GetBindingName().
 */
inline std::string ProgramCall(const std::string& programName)
{
  std::ostringstream oss;
  std::string command_prefix = "R> ";
  oss << command_prefix;

  // Determine if we have any output options.
  std::map<std::string, util::ParamData>& parameters = IO::Parameters();
  bool hasOutput = false;
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    if (!it->second.input)
    {
      hasOutput = true;
      break;
    }
  }

  if (hasOutput)
    oss << "d <- ";

  oss << programName << "(";

  // Now iterate over every input option.
  bool first = true;
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    if (!it->second.input || (it->second.persistent &&
        it->second.name != "verbose"))
      continue;

    if (!first)
      oss << ", ";
    else
      first = false;

    // Print the input option.
    oss << it->second.name << "=";

    std::string value;
    IO::GetSingleton().functionMap[it->second.tname]["DefaultParam"](
        it->second, NULL, (void*) &value);
    oss << value;
  }
  oss << ")";

  std::string result = util::HyphenateString(oss.str(), 8);

  oss.str("");
  oss << result;

  // Now print output lines.
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    if (it->second.input)
      continue;

    // Print a new line for the output option.
    oss << std::endl << command_prefix << it->second.name << " <- d$"
        << it->second.name;
  }

  return oss.str();
}

/**
 * Given the name of a model, print it.  Here we do not need to modify anything.
 */
inline std::string PrintModel(const std::string& modelName)
{
  return "\"" + modelName + "\"";
}

/**
 * Given the name of a matrix, print it.  Here we do not need to modify
 * anything.
 */
inline std::string PrintDataset(const std::string& datasetName)
{
  return "\"" + datasetName + "\"";
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
  // For a R binding we don't need to know the type.
  return "\"" + paramName + "\"";
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
  return !IO::Parameters()[paramName].input;
}

inline bool IgnoreCheck(const std::vector<std::string>& constraints)
{
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!IO::Parameters()[constraints[i]].input)
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
    if (!IO::Parameters()[constraints[i].first].input)
      return true;
  }

  return !IO::Parameters()[paramName].input;
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
