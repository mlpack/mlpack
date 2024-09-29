/**
 * @file bindings/markdown/print_doc_functions.hpp
 * @author Ryan Curtin
 *
 * This file wraps the different printing functionality of different binding
 * types.  If a new binding type is added, this code will need to be modified so
 * that Markdown can be printed.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_MARKDOWN_PRINT_DOC_FUNCTIONS_HPP
#define MLPACK_BINDINGS_MARKDOWN_PRINT_DOC_FUNCTIONS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace markdown {

/**
 * Given the name of the binding, print the name for the current language (as
 * given by BindingInfo).
 */
inline std::string GetBindingName(const std::string& bindingName);

/**
 * Given the name of the binding, print the name for the wrapper for
 * current language.
 */
inline std::string GetWrapperName(const std::string& bindingName);

/**
 * Print the name of the given language.
 */
inline std::string PrintLanguage(const std::string& language);

/**
 * Print any imports that need to be done before using the binding.
 */
inline std::string PrintImport(const std::string& bindingName);

/**
 * Print any special information about input options.
 */
inline std::string PrintInputOptionInfo(const std::string& language);

/**
 * Print any special information about output options.
 */
inline std::string PrintOutputOptionInfo(const std::string& language);

/**
 * Print details about the different types for a language.
 */
inline std::string PrintTypeDocs();

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes);

/**
 * Print the default value of an option, unless it is required (in which case
 * Markdown italicized '--' is printed).
 */
inline std::string PrintDefault(const std::string& bindingName,
                                const std::string& paramName);

/**
 * Print a dataset type parameter (add .csv and return).
 */
inline std::string PrintDataset(const std::string& dataset);

/**
 * Print a model type parameter (add .bin and return).
 */
inline std::string PrintModel(const std::string& model);

/**
 * Given a program name and arguments for it, print what its invocation would
 * be.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args);

/**
 * Given a program name, print a call assuming that all arguments are specified.
 */
inline std::string ProgramCall(const std::string& programName);

/**
 * Print what a user would type to invoke the given option name.  Note that the
 * name *must* exist in the IO module.  (Note that because of the way
 * BINDING_LONG_DESC() and BINDING_EXAMPLE() is structured, this doesn't mean
 * that all of the PARAM_*() declarataions need to come before
 * BINDING_LONG_DESC() and BINDING_EXAMPLE() declaration.)
 */
inline std::string ParamString(const std::string& bindingName,
                               const std::string& paramName);

/**
 * Print the user-encountered type of an option.
 */
inline std::string ParamType(util::Params& p, util::ParamData& d);

/**
 * Print the import string that imports any external libs.
 */
inline std::string ImportExtLib();

/**
 * Print the import string that imports mlpack's preprocess_split
 * method.
 */
inline std::string ImportSplit();

/**
 * Import the current method.
 */
inline std::string ImportThis(const std::string& groupName);

/**
 * Print the string that splits dataset into training and testing.
 */
inline std::string SplitTrainTest(const std::string& datasetName,
                                  const std::string& labelName,
                                  const std::string& trainDataset,
                                  const std::string& trainLabels,
                                  const std::string& testDataset,
                                  const std::string& testLabels,
                                  const std::string& splitRatio);

/**
 * Print the string that reads dataset from an online source.
 */
inline std::string GetDataset(const std::string& datasetName,
                              const std::string& url);

/**
 * Print the string that creates object of the given method
 * with the given parameters.
 */
template<typename... Args>
std::string CreateObject(const std::string& bindingName,
                         const std::string& objectName,
                         const std::string& groupName,
                         Args... args);

/**
 * Print the string that creates object of the given method
 * with default parameters.
 */
inline std::string CreateObject(const std::string& bindingName,
                                const std::string& objectName,
                                const std::string& groupName);

/**
 * Print the string that calls a method from the object created.
 */
template<typename... Args>
std::string CallMethod(const std::string& bindingName,
                       const std::string& objectName,
                       const std::string& methodName,
                       Args... args);

/**
 * Get the mapped name in the corresponding language.
 */
inline std::string GetMappedName(const std::string& methodName);

inline std::string GetWrapperLink(const std::string& bindingName);

/**
 * Return whether or not a runtime check on parameters should be ignored.
 */
template<typename T>
inline bool IgnoreCheck(const std::string& bindingName, const T& t);

} // namespace markdown
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "print_doc_functions_impl.hpp"

#endif
