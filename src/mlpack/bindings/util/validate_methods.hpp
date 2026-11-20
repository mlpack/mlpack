/**
 * @file validate_methods.hpp
 * @author Ryan Curtin
 *
 * Ensure that the parameters for each of the individual methods of a grouped
 * binding satisfy a handful of requirements.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_UTIL_VALIDATE_METHODS_HPP
#define MLPACK_BINDINGS_UTIL_VALIDATE_METHODS_HPP

namespace mlpack {
namespace bindings {
namespace util {

/**
 * Check the validity of each of the methods of a grouped binding:
 *
 *  - The "train" binding should have two required parameters of matrix type,
 *    and output only one model parameter.
 *
 *  - A "predict"/"classify"/"probabilities" binding should have two required
 *    parameters, one of matrix type and one of model type, and output only one
 *    matrix parameter.
 *
 *  - All methods should have only one output parameter.
 */
inline void ValidateMethods(
    const std::vector<std::string>& methods,
    const std::map<std::string, mlpack::util::Params>& params,
    const std::string& callerName);

} // namespace util
} // namespace bindings
} // namespace mlpack

#include "validate_methods_impl.hpp"

#endif
