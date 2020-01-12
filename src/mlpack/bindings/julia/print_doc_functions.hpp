/**
 * @file print_doc_functions.hpp
 * @author Ryan Curtin
 *
 * This file contains functions useful for printing documentation strings
 * related to Julia bindings.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_PRINT_DOC_FUNCTIONS_HPP
#define MLPACK_BINDINGS_JULIA_PRINT_DOC_FUNCTIONS_HPP

#include <mlpack/core/util/hyphenate_string.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

/**
 * Given the name of a binding, print its Julia name (this just returns the
 * binding name).
 */
inline std::string GetBindingName(const std::string& bindingName);

/**
 * Print any imports for Julia.
 */
inline std::string PrintImport(const std::string& bindingName);

/**
 * Print any special information about output options.
 */
inline std::string PrintOutputOptionInfo();

/**
 * Print documentation for each of the types.
 */
inline std::string PrintTypeDocs();

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
inline std::string PrintDefault(const std::string& paramName);

/**
 * Print a dataset type parameter.
 */
inline std::string PrintDataset(const std::string& dataset);

/**
 * Print a model type parameter.
 */
inline std::string PrintModel(const std::string& model);

/**
 * Print the type of a parameter that a user would specify from Julia.
 */
inline std::string PrintType(const util::ParamData& param);

// Recursion base case.
inline std::string PrintInputOptions();

/**
 * Print an input option.  This will throw an exception if the parameter does
 * not exist in CLI.  For a parameter 'x' with value '5', this will print
 * something like x=5.
 */
template<typename T, typename... Args>
std::string PrintInputOptions(const std::string& paramName,
                              const T& value,
                              Args... args);

// Recursion base case.
inline std::string PrintOutputOptions();

template<typename T, typename... Args>
std::string PrintOutputOptions(const std::string& paramName,
                               const T& value,
                               Args... args);

/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args);

/**
 * Given the parameter name, determine what it would actually be when passed to
 * the command line.
 */
inline std::string ParamString(const std::string& paramName);

/**
 * Print whether or not we should ignore a check on the given parameter.  For
 * Julia bindings, we ignore any checks on output parameters, so if paramName
 * is an output parameter, this returns true.
 */
inline bool IgnoreCheck(const std::string& paramName);

/**
 * Print whether or not we should ignore a check on the given set of
 * constraints.  For Julia bindings, we ignore any checks on output parameters,
 * so if any parameter is an output parameter, this returns true.
 */
inline bool IgnoreCheck(const std::vector<std::string>& constraints);

/**
 * Print whether or not we should ignore a check on the given set of
 * constraints.  For Julia bindings, we ignore any checks on output parameters,
 * so if any constraint parameter or the main parameter are output parameters,
 * this returns true.
 */
inline bool IgnoreCheck(
    const std::vector<std::pair<std::string, bool>>& constraints,
    const std::string& paramName);

} // namespace julia
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "print_doc_functions_impl.hpp"

#endif
