/**
 * @file backtrace.hpp
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
  /**
   * Constructor initialize fields and call GetAddress to retrieve addresses
   * for each frame of backtrace.
   *
   * @param maxDepth Maximum depth of backtrace. Default 32 steps.
   */
#ifdef HAS_BFD_DL
  Backtrace(int maxDepth = 32);
#else
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
  static void GetAddress(int maxDepth);

  /**
   * Decodes file name, function & line number.
   *
   * @param address Address of traced frame.
   */
  static void DecodeAddress(long address);

  //! Demangles function name.
  static void DemangleFunction();

  //! Backtrace datastructure.
  struct Frames
  {
    void *address;
    const char* function;
    const char* file;
    unsigned line;
  } static frame;

  //! A vector for all the backtrace information.
  static std::vector<Frames> stack;
};

}; //namespace mlpack

#endif
