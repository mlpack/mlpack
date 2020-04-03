#ifndef MLPACK_BINDINGS_JAVA_PRINT_DOC_FUNCTIONS
#define MLPACK_BINDINGS_JAVA_PRINT_DOC_FUNCTIONS

#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace bindings {
namespace java {

/**
 * Given the parameter name, determine what it would actually be when passed to
 * the command line.
 */
inline std::string ParamString(const std::string& paramName);

/**
 * Given a parameter type, print the corresponding value.
 */
template<typename T>
inline std::string PrintValue(const T& value, bool quotes);

// Special overload for booleans.
template<>
inline std::string PrintValue(const bool& value, bool quotes);

/**
 * Given the name of a matrix, print it.  Here we do not need to modify
 * anything.
 */
inline std::string PrintDataset(const std::string& datasetName);

/**
 * Given the name of a model, print it.  Here we do not need to modify anything.
 */
inline std::string PrintModel(const std::string& modelName);

/**
 * Given a name of a binding and a variable number of arguments (and their
 * contents), print the corresponding function call.
 */
template<typename... Args>
std::string ProgramCall(const std::string& programName, Args... args);

/**
 * Given the name of a binding, print a program call assuming that all options
 * are specified.
 */
inline std::string ProgramCall(const std::string& programName);

/**
 * Print whether or not we should ignore a check on the given parameter.  For
 * Python bindings, we ignore any checks on output parameters, so if paramName
 * is an output parameter, this returns true.
 */
inline bool IgnoreCheck(const std::string& paramName);

/**
 * Print whether or not we should ignore a check on the given set of
 * constraints.  For Python bindings, we ignore any checks on output parameters,
 * so if any parameter is an output parameter, this returns true.
 */
inline bool IgnoreCheck(const std::vector<std::string>& constraints);

/**
 * Print whether or not we should ignore a check on the given set of
 * constraints.  For Python bindings, we ignore any checks on output parameters,
 * so if any constraint parameter or the main parameter are output parameters,
 * this returns true.
 */
inline bool IgnoreCheck(
    const std::vector<std::pair<std::string, bool>>& constraints,
    const std::string& paramName);

} // namespace java
} // namespace bindings
} // namespace mlpack

#include "print_doc_functions_impl.hpp"

#endif
