/**
 * @file log.hpp
 * @author Matthew Amidon
 *
 * Definition of the Log class.
 *
 * This file is part of mlpack 2.0.0.
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
#ifndef __MLPACK_CORE_UTIL_LOG_HPP
#define __MLPACK_CORE_UTIL_LOG_HPP

#include <string>

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


  // We only use PrefixedOutStream if the program is compiled with debug
  // symbols.
#ifdef DEBUG
  //! Prints debug output with the appropriate tag: [DEBUG].
  static util::PrefixedOutStream Debug;
#else
  //! Dumps debug output into the bit nether regions.
  static util::NullOutStream Debug;
#endif

  //! Prints informational messages if --verbose is specified, prefixed with
  //! [INFO ].
  static util::PrefixedOutStream Info;

  //! Prints warning messages prefixed with [WARN ].
  static util::PrefixedOutStream Warn;

  //! Prints fatal messages prefixed with [FATAL], then terminates the program.
  static util::PrefixedOutStream Fatal;

  //! Reference to cout, if necessary.
  static std::ostream& cout;
};

}; //namespace mlpack

#endif
