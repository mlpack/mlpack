/**
 * @file print_doc_functions.hpp
 * @author Ryan Curtin
 *
 * This will generate a string representing what a user should type to invoke a
 * given option.  For the command-line bindings, this will generate strings like
 * '--param_name=x' or '--param_name'.
 */
#ifndef MLPACK_BINDINGS_CLI_PRINT_DOC_FUNCTIONS_HPP
#define MLPACK_BINDINGS_CLI_PRINT_DOC_FUCNTIONS_HPP

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

} // namespace cli
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "print_doc_functions_impl.hpp"

#endif
