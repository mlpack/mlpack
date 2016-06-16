/**
 * @file log.cpp
 * @author Matthew Amidon
 *
 * Implementation of the Log class.
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
#include "log.hpp"

#ifdef HAS_BFD_DL
  #include "backtrace.hpp"
#endif

// Color code escape sequences -- but not on Windows.
#ifndef _WIN32
  #define BASH_RED "\033[0;31m"
  #define BASH_GREEN "\033[0;32m"
  #define BASH_YELLOW "\033[0;33m"
  #define BASH_CYAN "\033[0;36m"
  #define BASH_CLEAR "\033[0m"
#else
  #define BASH_RED ""
  #define BASH_GREEN ""
  #define BASH_YELLOW ""
  #define BASH_CYAN ""
  #define BASH_CLEAR ""
#endif

using namespace mlpack;
using namespace mlpack::util;

// Only output debugging output if in debug mode.
#ifdef DEBUG
PrefixedOutStream Log::Debug = PrefixedOutStream(std::cout,
    BASH_CYAN "[DEBUG] " BASH_CLEAR);
#else
NullOutStream Log::Debug = NullOutStream();
#endif

PrefixedOutStream Log::Info = PrefixedOutStream(std::cout,
    BASH_GREEN "[INFO ] " BASH_CLEAR, true /* unless --verbose */, false);
PrefixedOutStream Log::Warn = PrefixedOutStream(std::cout,
    BASH_YELLOW "[WARN ] " BASH_CLEAR, false, false);
PrefixedOutStream Log::Fatal = PrefixedOutStream(std::cerr,
    BASH_RED "[FATAL] " BASH_CLEAR, false, true /* fatal */);

std::ostream& Log::cout = std::cout;

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
