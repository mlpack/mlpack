/**
 * @file bindings/python/wrapper_functions.hpp
 * @author Dirk Eddelbuettel
 *
 * Contains some utility functions for wrapper generation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_WRAPPER_FUNCTIONS_HPP
#define MLPACK_BINDINGS_R_WRAPPER_FUNCTIONS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace r {

// Get mapped name of an internal mlpack method.
inline std::string GetMappedName(const std::string& methodName);

} // namespace r
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "wrapper_functions_impl.hpp"

#endif
