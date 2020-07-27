/**
 * @file bindings/R/print_doc_functions.hpp
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
#ifndef MLPACK_BINDINGS_R_PRINT_DOC_FUNCTIONS_HPP
#define MLPACK_BINDINGS_R_PRINT_DOC_FUNCTIONS_HPP

#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace r {

/**
 * Given the name of a binding, print its R name.
 */
inline std::string GetBindingName(const std::string& bindingName);

/**
 * Print any import information for the R binding.
 */
inline std::string PrintImport();

/**
 * Print any special information about input options.
 */
inline std::string PrintInputOptionInfo();

/**
 * Print any special information about output options.
 */
inline std::string PrintOutputOptionInfo();

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes);

/**
 * Special overload for booleans.
 */
template<>
inline std::string PrintValue(const bool& value, bool quotes);

/**
 * Given a parameter name, print its corresponding default value.
 */
inline std::string PrintDefault(const std::string& paramName);

/**
 * Recursion base case.
 */
inline std::string PrintInputOptions();

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in IO.
 */
template<typename T, typename... Args>
std::string PrintInputOptions(const std::string& paramName,
                              const T& value,
                              Args... args);

/**
 * Recursion base case.
 */
inline std::string PrintOutputOptions(const bool /* markdown */);

template<typename T, typename... Args>
std::string PrintOutputOptions(const bool markdown,
                               const std::string& paramName,
                               const T& value,
                               Args... args);

/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
std::string ProgramCall(const bool markdown,
                        const std::string& programName,
                        Args... args);

/**
 * Given the name of a binding, print a program call assuming that all options
 * are specified.
 */
inline std::string ProgramCall(const std::string& programName);

/**
 * Given the name of a model, print it.  Here we do not need to modify anything.
 */
inline std::string PrintModel(const std::string& modelName);

/**
 * Given the name of a matrix, print it.  Here we do not need to modify
 * anything.
 */
inline std::string PrintDataset(const std::string& datasetName);

/**
 * Given the parameter name, determine what it would actually be when passed to
 * the command line.
 */
inline std::string ParamString(const std::string& paramName);

/**
 * Print whether or not we should ignore a check on the given parameter.
 */
inline bool IgnoreCheck(const std::string& paramName);

/**
 * Print whether or not we should ignore a check on the given set of
 * constraints.
 */
inline bool IgnoreCheck(const std::vector<std::string>& constraints);

/**
 * Print whether or not we should ignore a check on the given set of
 * constraints.
 */
inline bool IgnoreCheck(
    const std::vector<std::pair<std::string, bool>>& constraints,
    const std::string& paramName);

} // namespace r
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "print_doc_functions_impl.hpp"

#endif
