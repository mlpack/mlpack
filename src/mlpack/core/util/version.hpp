/**
 * @file version.hpp
 * @author Ryan Curtin
 *
 * The current version of mlpack, available as macros and as a string.
 */
#ifndef MLPACK_CORE_UTIL_VERSION_HPP
#define MLPACK_CORE_UTIL_VERSION_HPP

#include <string>

// The version of mlpack.  If this is a git repository, this will be a version
// with higher number than the most recent release.
#define MLPACK_VERSION_MAJOR 2
#define MLPACK_VERSION_MINOR 0
#define MLPACK_VERSION_PATCH "x"

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
