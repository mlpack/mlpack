/**
 * @file log.cpp
 * @author Matthew Amidon
 *
 * Implementation of the Log class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include "log.hpp"

#ifdef HAS_BFD_DL
  #include "backtrace.hpp"
#endif

using namespace mlpack;
using namespace mlpack::util;

// Only do anything for Assert() if in debugging mode.
#ifdef DEBUG
void Log::Assert(bool condition, const std::string& message)
{
  if (!condition)
  {
#ifdef HAS_BFD_DL
    Backtrace bt;

    Log::Debug << bt.ToString();
#endif
    Log::Debug << message << std::endl;

    throw std::runtime_error("Log::Assert() failed: " + message);
  }
}
#else
void Log::Assert(bool /* condition */, const std::string& /* message */)
{ }
#endif
