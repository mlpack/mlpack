/**
 * @file log.cpp
 * @author Matthew Amidon
 *
 * Implementation of the Log class.
 */
#include <cxxabi.h>
#include <execinfo.h>

#include "log.hpp"

// Color code escape sequences.
#define BASH_RED "\033[0;31m"
#define BASH_GREEN "\033[0;32m"
#define BASH_YELLOW "\033[0;33m"
#define BASH_CYAN "\033[0;36m"
#define BASH_CLEAR "\033[0m"

using namespace mlpack;
using namespace mlpack::io;

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
void Log::Assert(bool condition, const char* message)
{
  if(!condition)
  {
    void* array[25];
    size_t size = backtrace (array, sizeof(array)/sizeof(void*));
    char** messages = backtrace_symbols(array, size);

    // skip first stack frame (points here)
    for (size_t i = 1; i < size && messages != NULL; ++i)
    {
      char *mangled_name = 0, *offset_begin = 0, *offset_end = 0;

      // find parantheses and +address offset surrounding mangled name
      for (char *p = messages[i]; *p; ++p)
      {
        if (*p == '(')
        {
          mangled_name = p;
        }
        else if (*p == '+')
        {
          offset_begin = p;
        }
        else if (*p == ')')
        {
          offset_end = p;
          break;
        }
      }

      // if the line could be processed, attempt to demangle the symbol
      if (mangled_name && offset_begin && offset_end &&
          mangled_name < offset_begin)
      {
        *mangled_name++ = '\0';
        *offset_begin++ = '\0';
        *offset_end++ = '\0';

        int status;
        char* real_name = abi::__cxa_demangle(mangled_name, 0, 0, &status);

        // if demangling is successful, output the demangled function name
        if (status == 0)
        {
          Log::Debug << "[bt]: (" << i << ") " << messages[i] << " : "
                    << real_name << "+" << offset_begin << offset_end
                    << std::endl;

        }
        // otherwise, output the mangled function name
        else
        {
          Log::Debug << "[bt]: (" << i << ") " << messages[i] << " : "
                    << mangled_name << "+" << offset_begin << offset_end
                    << std::endl;
        }
        free(real_name);
      }
      // otherwise, print the whole line
      else
      {
          Log::Debug << "[bt]: (" << i << ") " << messages[i] << std::endl;
      }
    }
    Log::Debug << message << std::endl;
    free(messages);

    //backtrace_symbols_fd (array, size, 2);
    exit(1);
  }
}
#else
void Log::Assert(bool condition, const char* message)
{ }
#endif
