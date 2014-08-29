/**
 * @file version.hpp
 * @author Ryan Curtin
 *
 * The current version of mlpack, available as macros and as a string.
 *
 * This file is part of MLPACK 1.0.10.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef __MLPACK_CORE_UTIL_VERSION_HPP
#define __MLPACK_CORE_UTIL_VERSION_HPP

#include <string>

// The version of mlpack.  If this is svn trunk, this will be a version with
// higher number than the most recent release.
#define __MLPACK_VERSION_MAJOR 1
#define __MLPACK_VERSION_MINOR 0
#define __MLPACK_VERSION_PATCH 9

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
