/**
 * @file version.cpp
 * @author Ryan Curtin
 *
 * The implementation of GetVersion().
 *
 * This file is part of mlpack 1.0.12.
 *
 * mlpack is free software; you may redstribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
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
