/**
 * @file bindings/julia/wrapper_functions.hpp
 * @author Ryan Curtin
 *
 * Contains some important utility functions for wrapper generation for Julia.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_JULIA_WRAPPER_FUNCTIONS_HPP
#define MLPACK_BINDINGS_JULIA_WRAPPER_FUNCTIONS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace julia {

// Get the name of the Julia class using the group name.
inline std::string GetClassName(const std::string& groupName);

// Get the valid name of a parameter (to avoid clashes with Julia keywords).
inline std::string GetValidName(const std::string& paramName);

// Get mapped name of an internal mlpack method.
inline std::string GetMappedName(const std::string& methodName);

} // namespace julia
} // namespace bindings
} // namespace mlpack

// Include implementation.
#include "wrapper_functions_impl.hpp"

#endif
