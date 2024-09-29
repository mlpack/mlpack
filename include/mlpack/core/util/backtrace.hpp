/**
 * @file core/util/backtrace.hpp
 * @author Grzegorz Krajewski
 *
 * Definition of the Backtrace class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef __MLPACK_CORE_UTIL_BACKTRACE_HPP
#define __MLPACK_CORE_UTIL_BACKTRACE_HPP

#include <string>
#include <vector>

#ifdef MLPACK_HAS_BFD_DL
  #include <execinfo.h>
  #include <signal.h>
  #include <unistd.h>
  #include <cxxabi.h>

  // Some versions of libbfd require PACKAGE and PACKAGE_VERSION to be set in
  // order for the include to not fail.  For more information:
  // https://github.com/mlpack/mlpack/issues/574
  #ifndef PACKAGE
    #define PACKAGE
    #ifndef PACKAGE_VERSION
      #define PACKAGE_VERSION
      #include <bfd.h>
      #undef PACKAGE_VERSION
    #else
      #include <bfd.h>
    #endif
    #undef PACKAGE
  #else
    #ifndef PACKAGE_VERSION
      #define PACKAGE_VERSION
      #include <bfd.h>
      #undef PACKAGE_VERSION
    #else
      #include <bfd.h>
    #endif
  #endif
  #include <dlfcn.h>
#endif

namespace mlpack {

/**
 * Provides a backtrace.
 *
 * The Backtrace class retrieve addresses of each called function from the
 * stack and decode file name, function & line number. Retrieved information
 * can be printed in form:
 *
 * @code
 * [b]: (count) /directory/to/file.cpp:function(args):line_number
 * @endcode
 *
 * Backtrace is printed always when Log::Assert failed.
 * An example is given below.
 *
 * @code
 * if (!someImportantCondition())
 * {
 *   Log::Fatal << "someImportantCondition() is not satisfied! Terminating.";
 *   Log::Fatal << std::endl;
 * }
 * @endcode
 *
 * @note Log::Assert will not be shown when compiling in non-debug mode.
 *
 * @see PrefixedOutStream, Log
 */
class Backtrace
{
 public:
#ifdef MLPACK_HAS_BFD_DL
  /**
   * Constructor initialize fields and call GetAddress to retrieve addresses
   * for each frame of backtrace.
   *
   * @param maxDepth Maximum depth of backtrace. Default 32 steps.
   */
  Backtrace(int maxDepth = 32);
#else
  /**
   * Constructor initialize fields and call GetAddress to retrieve addresses
   * for each frame of backtrace.
   *
   */
  Backtrace();
#endif
  //! Returns string of backtrace.
  std::string ToString();

 private:
  /**
   * Gets addresses of each called function from the stack.
   *
   * @param maxDepth Maximum depth of backtrace. Default 32 steps.
   */
  void GetAddress(int maxDepth);

  /**
   * Decodes file name, function & line number.
   *
   * @param address Address of traced frame.
   */
  void DecodeAddress(long address);

  //! Demangles function name.
  void DemangleFunction();

  //! Backtrace datastructure.
  struct Frames
  {
    void *address;
    const char* function;
    const char* file;
    unsigned line;
  };

  Frames frame;
  std::vector<Frames> stack;

#ifdef MLPACK_HAS_BFD_DL
  // Binary File Descriptor objects.
  bfd* abfd;          // Descriptor datastructure.
  asymbol **syms;     // Symbols datastructure.
  asection *text;     // Strings datastructure.
#endif
};

}; // namespace mlpack

// Include implementation.
#include "backtrace_impl.hpp"

#endif
