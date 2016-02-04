/**
 * @file version.hpp
 * @author Ryan Curtin
 *
 * The current version of mlpack, available as macros and as a string.
 *
 * This file is part of mlpack 2.0.0.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_UTIL_VERSION_HPP
#define __MLPACK_CORE_UTIL_VERSION_HPP

#include <string>

// The version of mlpack.  If this is a git repository, this will be a version
// with higher number than the most recent release.
#define __MLPACK_VERSION_MAJOR 1
#define __MLPACK_VERSION_MINOR "x"
#define __MLPACK_VERSION_PATCH "x"

// The name of the version (for use by --version).
namespace mlpack {
namespace util {

/**
 * This will return either "mlpack x.y.z" or "mlpack master-XXXXXXX" depending on
 * whether or not this is a stable version of mlpack or a git repository.
 */
std::string GetVersion();

} // namespace util
} // namespace mlpack

#endif
