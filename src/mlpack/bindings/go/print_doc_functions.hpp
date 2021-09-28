/**
 * @file bindings/go/print_doc_functions.hpp
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
#ifndef MLPACK_BINDINGS_GO_PRINT_DOC_FUNCTIONS_HPP
#define MLPACK_BINDINGS_GO_PRINT_DOC_FUNCTIONS_HPP

#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace go {

/**
 * Given the name of a binding, print its Go name.
 */
inline std::string GetBindingName(const std::string& bindingName);

/**
 * Print any import information for the Go binding.
 */
inline std::string PrintImport();

/**
 * Print any special information about output options.
 */
inline std::string PrintOutputOptionInfo();

/**
 * Print any special information about input options.
 */
inline std::string PrintInputOptionInfo();

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes);

// Special overload for booleans.
template<>
inline std::string PrintValue(const bool& value, bool quotes);

/**
 * Given a parameter name, print its corresponding default value.
 */
inline std::string PrintDefault(const std::string& bindingName,
                                const std::string& paramName);

// Base case: no modification needed.
inline void GetOptions(
    util::Params& /* params */,
    std::vector<std::tuple<std::string, std::string>>& /* results */);

/**
 * Assemble a vector of string tuples indicating parameter names and what should
 * be printed for them.  (For output parameters, we just need to print the
 * value.)
 */
template<typename T, typename... Args>
void GetOptions(
    util::Params& params,
    std::vector<std::tuple<std::string, std::string>>& results,
    const std::string& paramName,
    const T& value,
    Args... args);

// Recursion base case.
inline std::string PrintOptionalInputs(util::Params& /* params */);

// Recursion base case.
inline std::string PrintInputOptions(util::Params& /* params */);

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in IO.
 */
template<typename T, typename... Args>
std::string PrintOptionalInputs(util::Params& params,
                                const std::string& paramName,
                                const T& value,
                                Args... args);

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in IO.
 */
template<typename T, typename... Args>
std::string PrintInputOptions(util::Params& params,
                              const std::string& paramName,
                              const T& value,
                              Args... args);

// Recursion base case.
inline std::string PrintOutputOptions(util::Params& /* params */);

template<typename... Args>
std::string PrintOutputOptions(util::Params& params, Args... args);

/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args);

/**
 * Given the name of a binding, print its invocation.
 */
inline std::string ProgramCall(util::Params& params,
                               const std::string& programName);

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

} // namespace go
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "print_doc_functions_impl.hpp"

#endif
