/**
 * @file version.hpp
 * @author Ryan Curtin
 *
 * The current version of mlpack, available as macros and as a string.
 */
#ifndef __MLPACK_CORE_UTIL_VERSION_HPP
#define __MLPACK_CORE_UTIL_VERSION_HPP

#include <string>

// The version of mlpack.  If this is svn trunk, this will be a version with
// higher number than the most recent release.
#define __MLPACK_VERSION_MAJOR 1
#define __MLPACK_VERSION_MINOR x
#define __MLPACK_VERSION_PATCH x

// The name of the version (for use by --version).
namespace mlpack {
namespace util {

/**
 * This will return either "mlpack x.y.z" or "mlpack trunk-rXXXXX" depending on
 * whether or not this is a stable version of mlpack or an svn revision.
 */
std::string GetVersion();

}; // namespace util
}; // namespace mlpack

#endif
