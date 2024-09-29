/**
 * @file core/util/log_impl.hpp
 * @author Matthew Amidon
 *
 * Implementation of the Log class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_LOG_IMPL_HPP
#define MLPACK_CORE_UTIL_LOG_IMPL_HPP

#include "log.hpp"

#ifdef MLPACK_HAS_BFD_DL
  #include "backtrace.hpp"
#endif

namespace mlpack {

// Only do anything for Assert() if in debugging mode.
#ifdef DEBUG
inline void Log::Assert(bool condition, const std::string& message)
{
  if (!condition)
  {
#ifdef MLPACK_HAS_BFD_DL
    Backtrace bt;

    Log::Debug << bt.ToString();
#endif
    Log::Debug << message << std::endl;

    throw std::runtime_error("Log::Assert() failed: " + message);
  }
}
#else
inline void Log::Assert(bool /* condition */, const std::string& /* message */)
{ }
#endif

} // namespace mlpack

#endif
