/**
 * @file log.cpp
 * @author Matthew Amidon
 *
 * Implementation of the Log class.
 *
 * This file is part of MLPACK 1.0.5.
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
#ifndef _WIN32
  #include <cxxabi.h>
  #include <execinfo.h>
#endif

#include "log.hpp"

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
#ifndef _WIN32
    void* array[25];
    size_t size = backtrace (array, sizeof(array)/sizeof(void*));
    char** messages = backtrace_symbols(array, size);

    // skip first stack frame (points here)
    for (size_t i = 1; i < size && messages != NULL; ++i)
    {
      char *mangledName = 0, *offsetBegin = 0, *offsetEnd = 0;

      // find parantheses and +address offset surrounding mangled name
      for (char *p = messages[i]; *p; ++p)
      {
        if (*p == '(')
        {
          mangledName = p;
        }
        else if (*p == '+')
        {
          offsetBegin = p;
        }
        else if (*p == ')')
        {
          offsetEnd = p;
          break;
        }
      }

      // if the line could be processed, attempt to demangle the symbol
      if (mangledName && offsetBegin && offsetEnd &&
          mangledName < offsetBegin)
      {
        *mangledName++ = '\0';
        *offsetBegin++ = '\0';
        *offsetEnd++ = '\0';

        int status;
        char* realName = abi::__cxa_demangle(mangledName, 0, 0, &status);

        // if demangling is successful, output the demangled function name
        if (status == 0)
        {
          Log::Debug << "[bt]: (" << i << ") " << messages[i] << " : "
                    << realName << "+" << offsetBegin << offsetEnd
                    << std::endl;

        }
        // otherwise, output the mangled function name
        else
        {
          Log::Debug << "[bt]: (" << i << ") " << messages[i] << " : "
                    << mangledName << "+" << offsetBegin << offsetEnd
                    << std::endl;
        }
        free(realName);
      }
      // otherwise, print the whole line
      else
      {
          Log::Debug << "[bt]: (" << i << ") " << messages[i] << std::endl;
      }
    }
#endif
    Log::Debug << message << std::endl;

#ifndef _WIN32
    free(messages);
#endif

    //backtrace_symbols_fd (array, size, 2);
    exit(1);
  }
}
#else
void Log::Assert(bool /* condition */, const std::string& /* message */)
{ }
#endif
