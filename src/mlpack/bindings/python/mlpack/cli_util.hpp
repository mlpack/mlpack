/**
 * @file cli_util.hpp
 * @author Ryan Curtin
 *
 * Simple function to work around Cython's lack of support for lvalue
 * references.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_CYTHON_CLI_UTIL_HPP
#define MLPACK_BINDINGS_PYTHON_CYTHON_CLI_UTIL_HPP

#include <mlpack/core/util/cli.hpp>

namespace mlpack {
namespace util {

/**
 * Set the parameter to the given value.
 *
 * This function exists to work around Cython's lack of support for lvalue
 * references.
 *
 * @param identifier Name of parameter.
 * @param value Value to set parameter to.
 */
template<typename T>
inline void SetParam(const std::string& identifier, const T& value)
{
  CLI::GetParam<T>(identifier) = value;
}

/**
 * Turn verbose output on.
 */
inline void EnableVerbose()
{
  Log::Info.ignoreInput = false;
}

} // namespace util
} // namespace mlpack

#endif
