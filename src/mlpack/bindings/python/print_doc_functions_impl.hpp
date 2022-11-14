/**
 * @file bindings/python/print_doc_functions_impl.hpp
 * @author Ryan Curtin
 *
 * This file contains functions useful for printing documentation strings
 * related to Python bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_PRINT_DOC_FUNCTIONS_IMPL_HPP
#define MLPACK_BINDINGS_PYTHON_PRINT_DOC_FUNCTIONS_IMPL_HPP

#include <mlpack/core/util/hyphenate_string.hpp>
#include "wrapper_functions.hpp"

namespace mlpack {
namespace bindings {
namespace python {

/**
 * Given the name of a binding, print its Python name.
 */
inline std::string GetBindingName(const std::string& bindingName)
{
  // No modification is needed to the name---we just use it as-is.
  return bindingName + "()";
}

/**
 * Print any import information for the Python binding.
 */
inline std::string PrintImport(const std::string& bindingName)
{
  return "from mlpack import " + bindingName;
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
  return "Results are returned in a Python dictionary.  The keys of the "
      "dictionary are the names of the output parameters.";
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
  oss << "[";
  if (value.size() > 0)
  {
    oss << value[0];
    for (size_t i = 1; i < value.size(); ++i)
      oss << ", " << value[i];
  }
  oss << "]";
  if (quotes)
    oss << "'";
  return oss.str();
}

// Special overload for booleans.
template<>
inline std::string PrintValue(const bool& value, bool quotes)
{
  if (quotes && value)
    return "'True'";
  else if (quotes && !value)
    return "'False'";
  else if (!quotes && value)
    return "True";
  else
    return "False";
}

/**
 * Given a parameter name, print its corresponding default value.
 */
inline std::string PrintDefault(const std::string& bindingName,
                                const std::string& paramName)
{
  util::Params params = IO::Parameters(bindingName);

  if (params.Parameters().count(paramName) == 0)
    throw std::invalid_argument("unknown parameter " + paramName + "!");

  util::ParamData& d = params.Parameters()[paramName];

  std::string defaultValue;
  params.functionMap[d.tname]["DefaultParam"](d, NULL,
      (void*) &defaultValue);

  return defaultValue;
}

// Recursion base case.
std::string PrintInputOptions(util::Params& /* params */,
                              bool /* onlyHyperParams */,
                              bool /* onlyMatrixParams */)
{ return ""; }

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in IO.  For a parameter 'x' with value '5', this will print
 * something like x=5.
 */
template<typename T, typename... Args>
std::string PrintInputOptions(util::Params& params,
                              bool onlyHyperParams,
                              bool onlyMatrixParams,
                              const std::string& paramName,
                              const T& value,
                              Args... args)
{
  // See if this is part of the program.
  std::string result = "";
  if (params.Parameters().count(paramName) > 0)
  {
    util::ParamData& d = params.Parameters()[paramName];

    bool isSerial;
    params.functionMap[d.tname]["IsSerializable"](
        d, NULL, (void*) &isSerial);

    bool isHyperParam = false;
    size_t foundArma = d.cppType.find("arma");
    if(d.input && foundArma == std::string::npos &&
        !isSerial)
      isHyperParam = true;

    bool printCondition = d.input;

    // no parameter is both a hyper-parameter and a matrix-parmeter
    // hence the print condition is "false".
    if(onlyHyperParams && onlyMatrixParams) printCondition = false;
    else if(onlyHyperParams) printCondition = isHyperParam;
    else if(onlyMatrixParams) printCondition = foundArma != std::string::npos;

    if (printCondition)
    {
      // Print the input option.
      std::ostringstream oss;
      oss << GetValidName(paramName) << "=";
      oss << PrintValue(value, d.tname == TYPENAME(std::string));
      result = oss.str();
    }
  }
  else
  {
    // Unknown parameter!
    throw std::runtime_error("Unknown parameter '" + GetValidName(paramName) + "' " +
        "encountered while assembling documentation!  Check BINDING_LONG_DESC()"
        + " and BINDING_EXAMPLE() declaration.");
  }

  // Continue recursion.
  std::string rest = PrintInputOptions(params, onlyHyperParams,
                                       onlyMatrixParams, args...);
  if (rest != "" && result != "")
    result += ", " + rest;
  else if (result == "")
    result = rest;

  return result;
}

// Recursion base case.
inline std::string PrintOutputOptions(util::Params& /* params */) { return ""; }

template<typename T, typename... Args>
std::string PrintOutputOptions(util::Params& params,
                               const std::string& paramName,
                               const T& value,
                               Args... args)
{
  // See if this is part of the program.
  std::string result = "";
  if (params.Parameters().count(paramName) > 0)
  {
    util::ParamData& d = params.Parameters()[paramName];
    if (!d.input)
    {
      // Print a new line for the output option.
      std::ostringstream oss;
      oss << ">>> " << value << " = output['" << paramName << "']";
      result = oss.str();
    }
  }
  else
  {
    // Unknown parameter!
    throw std::runtime_error("Unknown parameter '" + paramName + "' " +
        "encountered while assembling documentation!  Check BINDING_LONG_DESC()"
        + " and BINDING_EXAMPLE() declaration.");
  }

  // Continue recursion.
  std::string rest = PrintOutputOptions(params, args...);
  if (rest != "" && result != "")
    result += '\n';
  result += rest;

  return result;
}

/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.  The given bindingName
 * should not be the output of GetBindingName().
 */
template<typename... Args>
std::string ProgramCall(const std::string& bindingName, Args... args)
{
  util::Params params = IO::Parameters(bindingName);

  std::ostringstream oss;
  oss << ">>> ";

  // Find out if we have any output options first.
  std::ostringstream ossOutput;
  ossOutput << PrintOutputOptions(params, args...);
  if (ossOutput.str() != "")
    oss << "output = ";
  oss << bindingName << "(";

  // Now process each input option.
  oss << PrintInputOptions(params, false, false, args...);
  oss << ")";

  std::string call = oss.str();
  oss.str(""); // Reset it.

  // Now process each output option.
  oss << PrintOutputOptions(params, args...);
  if (oss.str() == "")
    return util::HyphenateString(call, 2);
  else
    return util::HyphenateString(call, 2) + "\n" + oss.str();
}

/**
 * Given the name of a binding, print a program call assuming that all options
 * are specified.  The bindingName should not be the output of GetBindingName().
 */
inline std::string ProgramCall(util::Params& params,
                               const std::string& bindingName)
{
  std::ostringstream oss;
  oss << ">>> ";

  // Determine if we have any output options.
  std::map<std::string, util::ParamData>& parameters = params.Parameters();
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
    oss << "d = ";

  oss << bindingName << "(";

  // Now iterate over every input option.
  bool first = true;
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    if (!it->second.input)
      continue;

    if (!first)
      oss << ", ";
    else
      first = false;

    // Print the input option.
    const std::string correctName = GetValidName(it->second.name);
    oss << correctName << "=";

    std::string value;
    params.functionMap[it->second.tname]["DefaultParam"](
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
    oss << std::endl << ">>> " << it->second.name << " = d['"
        << it->second.name << "']";
  }

  return oss.str();
}

/**
 * Given the name of a model, print it.  Here we do not need to modify anything.
 */
inline std::string PrintModel(const std::string& modelName)
{
  return "'" + modelName + "'";
}

/**
 * Given the name of a matrix, print it.  Here we do not need to modify
 * anything.
 */
inline std::string PrintDataset(const std::string& datasetName)
{
  return "'" + datasetName + "'";
}

/**
 * Print any closing call to a program.  For a Python binding this is a closing
 * brace.
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
  // For a Python binding we don't need to know the type.

  // Make sure that we don't print reserved keywords or builtin functions.
  const std::string correctName = GetValidName(paramName);
  return "'" + correctName + "'";
}

/**
 * Given the parameter name and an argument, return what should be written as
 * documentation when referencing that argument.
 */
template<typename T>
inline std::string ParamString(const std::string& paramName, const T& value)
{
  std::ostringstream oss;
  oss << GetValidName(paramName) << "=" << value;
  return oss.str();
}

inline std::string ImportExtLib()
{
  return ">>> import pandas as pd";
}

inline std::string ImportSplit()
{
  return ">>> from mlpack import preprocess_split";
}

inline std::string ImportThis(const std::string& groupName)
{
  return ">>> from mlpack import " + GetClassName(groupName);
}

inline std::string SplitTrainTest(const std::string& datasetName,
                                  const std::string& labelName,
                                  const std::string& trainDataset,
                                  const std::string& trainLabels,
                                  const std::string& testDataset,
                                  const std::string& testLabels,
                                  const std::string& splitRatio)
{
  std::string splitString = ">>> ";
  splitString += testDataset + ", " + testLabels + ", ";
  splitString += trainDataset + ", " + trainLabels;
  splitString += " = ";
  splitString += "preprocess_split(input_=" + datasetName + ", input_labels=";
  splitString += labelName + ", test_ratio=" + splitRatio + ")";
  return splitString;
}

inline std::string GetDataset(const std::string& datasetName,
                              const std::string& url)
{
  std::string readString = ">>> " + datasetName + " = ";
  readString += "pd.read_csv('" + url + "')";
  return readString;
}

template<typename... Args>
std::string CreateObject(const std::string& bindingName,
                         const std::string& objectName,
                         const std::string& groupName,
                         Args... args)
{
  util::Params params = IO::Parameters(bindingName);
  std::string createObj = ">>> ";
  createObj += objectName + " = " + GetClassName(groupName) + "(";
  createObj += PrintInputOptions(params, true, false, args...);
  createObj += ")";
  return util::HyphenateString(createObj, 2);
}

inline std::string CreateObject(const std::string& bindingName,
                                const std::string& objectName,
                                const std::string& groupName)
{
  util::Params params = IO::Parameters(bindingName);
  std::map<std::string, util::ParamData>& parameters = params.Parameters();

  std::string createObj = ">>> ";
  createObj += objectName + " = " + GetClassName(groupName) + "(";

  bool first = true;
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    bool isSerial;
    params.functionMap[it->second.tname]["IsSerializable"](
        it->second, NULL, (void*) &isSerial);

    bool isHyperParam = false;
    size_t foundArma = it->second.cppType.find("arma");
    if(it->second.input && foundArma == std::string::npos &&
        !isSerial)
      isHyperParam = true;

    if(!isHyperParam) continue;
    if(it->second.name == "help" || it->second.name == "info" ||
        it->second.name == "version")
        continue;

    if (!first)
      createObj += ", ";
    else
      first = false;

    // Print the input option.
    createObj += GetValidName(it->second.name) + "=";

    std::string value;
    params.functionMap[it->second.tname]["DefaultParam"](
        it->second, NULL, (void*) &value);
    createObj += value;
  }
  createObj += ")";
  return util::HyphenateString(createObj, 2);
}

template<typename... Args>
std::string CallMethod(const std::string& bindingName,
                       const std::string& objectName,
                       const std::string& methodName,
                       Args... args)
{
  util::Params params = IO::Parameters(bindingName);
  std::map<std::string, util::ParamData> parameters = params.Parameters();
  std::string callMethod = ">>> ";

  // find out if there are any output options.
  for(auto it=parameters.begin(); it!=parameters.end(); it++)
  {
    if(it->second.input) continue;
    callMethod += it->first + ", ";
  }
  if (callMethod != "")
    callMethod = callMethod.substr(0, callMethod.size()-2);
  callMethod += " = " + objectName + "." + GetMappedName(methodName) + "(";
  callMethod += PrintInputOptions(params, false, true, args...);
  callMethod += ")";
  return util::HyphenateString(callMethod, 2);
}

inline bool IgnoreCheck(const std::string& bindingName,
                        const std::string& paramName)
{
  util::Params params = IO::Parameters(bindingName);
  return !params.Parameters()[paramName].input;
}

inline bool IgnoreCheck(const std::string& bindingName,
                        const std::vector<std::string>& constraints)
{
  util::Params params = IO::Parameters(bindingName);

  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!params.Parameters()[constraints[i]].input)
      return true;
  }

  return false;
}

inline bool IgnoreCheck(
    const std::string& bindingName,
    const std::vector<std::pair<std::string, bool>>& constraints,
    const std::string& paramName)
{
  util::Params params = IO::Parameters(bindingName);

  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!params.Parameters()[constraints[i].first].input)
      return true;
  }

  return !params.Parameters()[paramName].input;
}

} // namespace python
} // namespace bindings
} // namespace mlpack

#endif
