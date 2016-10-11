/**
 * @file log.cpp
 * @author Matthew Amidon
 *
 * Implementation of the Log class.
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
