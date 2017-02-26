/**
 * @file log.hpp
 * @author Matthew Amidon
 *
 * Definition of the Log class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_LOG_HPP
#define MLPACK_CORE_UTIL_LOG_HPP

#include <string>
#include <mlpack/mlpack_export.hpp>

#include "prefixedoutstream.hpp"
#include "nulloutstream.hpp"

namespace mlpack {

/**
 * Provides a convenient way to give formatted output.
 *
 * The Log class has four members which can be used in the same way ostreams can
 * be used:
 *
 *  - Log::Debug
 *  - Log::Info
 *  - Log::Warn
 *  - Log::Fatal
 *
 * Each of these will prefix a tag to the output (for easy filtering), and the
 * fatal output will terminate the program when a newline is encountered.  An
 * example is given below.
 *
 * @code
 * Log::Info << "Checking a condition." << std::endl;
 * if (!someCondition())
 *   Log::Warn << "someCondition() is not satisfied!" << std::endl;
 * Log::Info << "Checking an important condition." << std::endl;
 * if (!someImportantCondition())
 * {
 *   Log::Fatal << "someImportantCondition() is not satisfied! Terminating.";
 *   Log::Fatal << std::endl;
 * }
 * @endcode
 *
 * Any messages sent to Log::Debug will not be shown when compiling in non-debug
 * mode.  Messages to Log::Info will only be shown when the --verbose flag is
 * given to the program (or rather, the CLI class).
 *
 * @see PrefixedOutStream, NullOutStream, CLI
 */
class Log
{
 public:
  /**
   * Checks if the specified condition is true.
   * If not, halts program execution and prints a custom error message.
   * Does nothing in non-debug mode.
   */
  static void Assert(bool condition,
                     const std::string& message = "Assert Failed.");

  /**
   * MLPACK_EXPORT is required for global variables, so that they are properly
   * exported by the Windows compiler.
   */

  // We only use PrefixedOutStream if the program is compiled with debug
  // symbols.
#ifdef DEBUG
  //! Prints debug output with the appropriate tag: [DEBUG].
  static MLPACK_EXPORT util::PrefixedOutStream Debug;
#else
  //! Dumps debug output into the bit nether regions.
  static MLPACK_EXPORT util::NullOutStream Debug;
#endif

  //! Prints informational messages if --verbose is specified, prefixed with
  //! [INFO ].
  static MLPACK_EXPORT util::PrefixedOutStream Info;

  //! Prints warning messages prefixed with [WARN ].
  static MLPACK_EXPORT util::PrefixedOutStream Warn;

  //! Prints fatal messages prefixed with [FATAL], then terminates the program.
  static MLPACK_EXPORT util::PrefixedOutStream Fatal;

  //! Reference to cout, if necessary.
  static std::ostream& cout;
};

}; //namespace mlpack

#endif
