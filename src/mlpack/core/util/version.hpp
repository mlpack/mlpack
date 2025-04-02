/**
 * @file core/util/version.hpp
 * @author Ryan Curtin
 *
 * The current version of mlpack, available as macros and as a string.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_VERSION_HPP
#define MLPACK_CORE_UTIL_VERSION_HPP

#include <string>

// The version of mlpack.  If this is a git repository, this will be a version
// with higher number than the most recent release.
#define MLPACK_VERSION_MAJOR 4
#define MLPACK_VERSION_MINOR 6
#define MLPACK_VERSION_PATCH 0

// The name of the version (for use by --version).
namespace mlpack {
namespace util {

/**
 * This will return either "mlpack x.y.z" or "mlpack master-XXXXXXX" depending on
 * whether or not this is a stable version of mlpack or a git repository.
 */
inline std::string GetVersion();

} // namespace util
} // namespace mlpack

// Include implementation.
#include "version_impl.hpp"

#endif
