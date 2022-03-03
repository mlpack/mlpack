/**
 * @file bindings/python/wrapper_functions.hpp
 * @author Nippun Sharma
 *
 * Contains some important utility functions for wrapper generation.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_WRAPPER_FUNCTIONS_HPP
#define MLPACK_BINDINGS_PYTHON_WRAPPER_FUNCTIONS_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace bindings {
namespace python {

// Get the name of python class using the group name.
inline std::string GetClassName(const std::string& groupName);

// Get the valid name of a parameter (to avoid clashes with python)
// keywords.
inline std::string GetValidName(const std::string& paramName);

// Get a std::vector of methods through a string seperated by ' '.
inline std::vector<std::string> GetMethods(const std::string& validMethods);

// Get mapped name of an internal mlapck method.
inline std::string GetMappedName(const std::string& methodName);

} // python.
} // bindings.
} // mlpack.

// Include implementation.
#include "wrapper_functions_impl.hpp"

#endif
