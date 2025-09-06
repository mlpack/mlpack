/**
 * @file core/util/version_impl.hpp
 * @author Ryan Curtin
 *
 * The implementation of GetVersion().
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "version.hpp"

#include <sstream>

namespace mlpack {
namespace util {

// If we are not a git revision, just use the macros to assemble the version
// name.
inline std::string GetVersion()
{
  std::stringstream o;
  o << "mlpack " << MLPACK_VERSION_MAJOR << "." << MLPACK_VERSION_MINOR
      << "." << MLPACK_VERSION_PATCH;
  #if defined(MLPACK_PRERELEASE)
  o << " (prerelease)";
  #endif
  return o.str();
}

} // namespace util
} // namespace mlpack
