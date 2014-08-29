/**
 * @file version.cpp
 * @author Ryan Curtin
 *
 * The implementation of GetVersion().
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
#include "version.hpp"

#include <sstream>

// If we are not an svn revision, just use the macros to assemble the version
// name.
std::string mlpack::util::GetVersion()
{
#ifndef __MLPACK_SUBVERSION
  std::stringstream o;
  o << "mlpack " << __MLPACK_VERSION_MAJOR << "." << __MLPACK_VERSION_MINOR
      << "." << __MLPACK_VERSION_PATCH;
  return o.str();
#else
  #include "svnversion.hpp"
#endif
}
