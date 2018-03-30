/**
 * @file print_doc_functions.hpp
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
#ifndef MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_HPP
#define MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_HPP

#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes);

/**
 * Print a dataset type parameter (add .csv and return).
 */
inline std::string PrintDataset(const std::string& dataset);

/**
 * Print a model type parameter (add .bin and return).
 */
inline std::string PrintModel(const std::string& model);

/**
 * Base case for recursion.
 */
inline std::string ProcessOptions();

/**
 * Print an option for a command-line argument.
 */
template<typename T, typename... Args>
std::string ProcessOptions(const std::string& paramName,
                           const T& value,
                           Args... args);

/**
 * Given a program name and arguments for it, print what its invocation would
 * be.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args);

/**
 * Print what a user would type to invoke the given option name.  Note that the
 * name *must* exist in the CLI module.  (Note that because of the way
 * ProgramInfo is structured, this doesn't mean that all of the PARAM_*()
 * declarataions need to come before the PROGRAM_INFO() declaration.)
 */
inline std::string ParamString(const std::string& paramName);

/**
 * Return whether or not a runtime check on parameters should be ignored.  We
 * don't ignore any runtime checks for CLI bindings, so this always returns
 * false.
 */
template<typename T>
inline bool IgnoreCheck(const T& /* t */) { return false; }

} // namespace cli
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "print_doc_functions_impl.hpp"

#endif
