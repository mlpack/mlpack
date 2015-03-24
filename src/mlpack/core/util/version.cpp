/**
 * @file version.cpp
 * @author Ryan Curtin
 *
 * The implementation of GetVersion().
 */
#include "version.hpp"

#include <sstream>

// If we are not a git revision, just use the macros to assemble the version
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
