/**
 * @file bindings/R/print_doc_functions_impl.hpp
 * @author Yashwant Singh Parihar
 * @author Dirk Eddelbuettel
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
inline std::string PrintDefault(const std::string& bindingName,
                                const std::string& paramName)
{
  util::Params p = IO::Parameters(bindingName);
  if (p.Parameters().count(paramName) == 0)
    throw std::invalid_argument("unknown parameter " + paramName + "!");

  util::ParamData& d = p.Parameters()[paramName];

  std::string defaultValue;
  p.functionMap[d.tname]["DefaultParam"](d, NULL, (void*) &defaultValue);

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
std::string PrintInputOptions(util::Params& /* p */)
{
  return "";
}

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in IO.
 */
template<typename T, typename... Args>
std::string PrintInputOptions(util::Params& p,
                              const std::string& paramName,
                              const T& value,
                              Args... args)
{
  // See if this is part of the program.
  std::string result = "";
  if (p.Parameters().count(paramName) > 0)
  {
    util::ParamData& d = p.Parameters()[paramName];
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
  std::string rest = PrintInputOptions(p, args...);
  if (rest != "" && result != "")
    result += ", " + rest;
  else if (result == "")
    result = rest;

  return result;
}

/**
 * Recursion base case.
 */
inline std::string PrintOutputOptions(util::Params& /* p */,
                                      const bool /* markdown */)
{
  return "";
}

template<typename T, typename... Args>
std::string PrintOutputOptions(util::Params& p,
                               const bool markdown,
                               const std::string& paramName,
                               const T& value,
                               Args... args)
{
  // See if this is part of the program.
  std::string result = "";
  std::string command_prefix = "R> ";
  if (p.Parameters().count(paramName) > 0)
  {
    util::ParamData& d = p.Parameters()[paramName];
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
  std::string rest = PrintOutputOptions(p, markdown, args...);
  if (rest != "" && result != "")
    result += "\n";
  result += rest;

  return result;
}

/**
 * Recursion base case.
 */
inline std::string GetOnlyOutputOptionName(util::Params&) { return ""; }

/**
 * Return an output name, called only in the case where we know only
 * one is available so we return early at first (and only) match.
 */
template<typename T, typename... Args>
std::string GetOnlyOutputOptionName(util::Params& p,
                                    const std::string& paramName,
                                    const T& value,
                                    Args... args)
{
  // Make sure the parameter exists.
  if (p.Parameters().count(paramName) > 0)
  {
    util::ParamData& d = p.Parameters()[paramName];
    if (!d.input)
    {
      // Since we already know we only have one output parameter,
      // we can just print this one and return entirely.
      std::ostringstream oss;
      oss << value;
      return oss.str();
    }
  }

  return GetOnlyOutputOptionName(p, args...);
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
  util::Params p = IO::Parameters(programName);
  std::ostringstream oss;
  if (markdown)
    oss << "R> ";

  size_t numOutputParams = 0;
  auto parameters = p.Parameters();
  for (auto it = parameters.begin(); it != parameters.end(); ++it)
  {
    if (!it->second.input)
      ++numOutputParams;
  }

  // Find out if we have any output options first.
  std::ostringstream ossOutput;
  ossOutput << PrintOutputOptions(p, markdown, args...);
  if (ossOutput.str() != "")
  {
    if (numOutputParams == 1)
      oss << GetOnlyOutputOptionName(p, args...) << " <- ";
    else
      oss << "output <- ";
  }
  oss << programName << "(";

  // Now process each input option.
  oss << PrintInputOptions(p, args...);
  oss << ")";

  std::string call = oss.str();
  oss.str(""); // Reset it.

  // Add lines to process output parameters, if there are more than one.
  // This generates lines like:
  //   predictions <- output$predictions
  if (numOutputParams > 1)
    oss << PrintOutputOptions(p, markdown, args...);

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
inline std::string ProgramCall(util::Params& p, const std::string& programName)
{
  std::ostringstream oss;
  std::string command_prefix = "R> ";
  oss << command_prefix;

  // Determine if we have any output options.
  std::map<std::string, util::ParamData>& parameters = p.Parameters();
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
    if (!it->second.input)
      continue;

    if (!first)
      oss << ", ";
    else
      first = false;

    // Print the input option.
    oss << it->second.name << "=";

    std::string value;
    p.functionMap[it->second.tname]["DefaultParam"](it->second, NULL,
        (void*) &value);
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

inline std::string ImportExtLib()
{
  // This function has to exist to satisfy the cross-language macro.
  // For R, we do no need anything here as no external libraries are loaded.
  return std::string("\\dontrun{\nsuppressMessages(library(data.table)) "
    "# for fread()");
}

inline std::string ImportSplit()
{
  // This function has to exist to satisfy the cross-language macro.
  // For R, we do no need anything here as no additional library are loaded.
  return "";
}

inline std::string ImportThis(const std::string& /* groupName */)
{
  // This function has to exist to satisfy the cross-language macro.
  // For R, we use it to load data.table for its fread() function
  return std::string("suppressMessages(library(mlpack)) "
    " # in case 'mlpack' is not yet loaded");
}

inline std::string GetDataset(const std::string& datasetName,
                              const std::string& url)
{
  return datasetName + " <- fread(\"" + url + "\", showProgress=FALSE)";
}

inline std::string SplitTrainTest(const std::string& datasetName,
                                  const std::string& labelName,
                                  const std::string& /* trainDataset */,
                                  const std::string& /* trainLabels */,
                                  const std::string& /* testDataset */,
                                  const std::string& /* testLabels */,
                                  const std::string& splitRatio)
{
  return std::string("pp <- preprocess_split(input=") + datasetName +
    ", input_label=as.matrix(1:nrow(" + datasetName + "))" +
    ", test_ratio=" + splitRatio + ")\n" +
    "X_train <- pp[[\"training\"]]\n" +
    "X_test <- pp[[\"test\"]]\n" +
    "# labels are indices to operate on both factors or numeric data\n" +
    "y_train <- " + labelName + "[as.integer(pp[[\"training_labels\"]]), 1]\n" +
    "y_test <- " + labelName + "[as.integer(pp[[\"test_labels\"]]), 1]";
}

template<typename... Args>
std::string CreateObject(const std::string& /* bindingName */,
                         const std::string& /* objectName */,
                         const std::string& /* groupName */,
                         Args... /* args */)
{
  return "";
}

inline std::string CreateObject(const std::string& /* bindingName */,
                                const std::string& /* objectName */,
                                const std::string& /* groupName */ )
{
  return "";
}

template<typename... Args>
std::string CallMethod(const std::string& bindingName,
                       const std::string& objectName,
                       const std::string& methodName,
                       Args... args)
{
  util::Params params = IO::Parameters(bindingName);
  std::map<std::string, util::ParamData> parameters = params.Parameters();
  std::string callMethod = "";

  if (methodName == "train")
  {
    callMethod += objectName;
    callMethod += " <- " + bindingName + "(";
  }
  else if (methodName == "classify" ||
           methodName == "predict" ||
           methodName == "probabilities")
  {
    callMethod += "\\dontrun{ ";
    callMethod += (methodName != "probabilities" ? "pred" : "prob");
    callMethod += " <- " +
      (methodName == "train" ? bindingName : "predict") +
      "(" + objectName + ", ";
  }
  callMethod += PrintInputOptions(params, args...);
  if (methodName == "probabilities")
    callMethod += ", type=\"probabilities\"";
  callMethod += ")";
  if (methodName == "train")
    callMethod += "\n";
  else
    callMethod += " ";
  callMethod += "}";
  return util::HyphenateString(callMethod, 2);
}



inline bool IgnoreCheck(const std::string& bindingName,
                        const std::string& paramName)
{
  util::Params p = IO::Parameters(bindingName);
  return !p.Parameters()[paramName].input;
}

inline bool IgnoreCheck(const std::string& bindingName,
                        const std::vector<std::string>& constraints)
{
  util::Params p = IO::Parameters(bindingName);
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!p.Parameters()[constraints[i]].input)
      return true;
  }

  return false;
}

inline bool IgnoreCheck(
    const std::string& bindingName,
    const std::vector<std::pair<std::string, bool>>& constraints,
    const std::string& paramName)
{
  util::Params p = IO::Parameters(bindingName);
  for (size_t i = 0; i < constraints.size(); ++i)
  {
    if (!p.Parameters()[constraints[i].first].input)
      return true;
  }

  return !p.Parameters()[paramName].input;
}

} // namespace r
} // namespace bindings
} // namespace mlpack

#endif
