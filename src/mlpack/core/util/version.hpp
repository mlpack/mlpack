/**
 * @file version.hpp
 * @author Ryan Curtin
 *
 * The current version of mlpack, available as macros and as a string.
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef MLPACK_CORE_UTIL_VERSION_HPP
#define MLPACK_CORE_UTIL_VERSION_HPP

#include <string>

// The version of mlpack.  If this is a git repository, this will be a version
// with higher number than the most recent release.
#define MLPACK_VERSION_MAJOR 2
#define MLPACK_VERSION_MINOR 0
#define MLPACK_VERSION_PATCH "x"

// Reverse compatibility; these macros will be removed in future versions of
// mlpack (3.0.0 and newer)!
#define __MLPACK_VERSION_MAJOR 2
#define __MLPACK_VERSION_MINOR 0
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
