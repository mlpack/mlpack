/**
 * @file option.hpp
 * @author Matthew Amidon
 *
 * Definition of the Option class, which is used to define parameters which are
 * used by CLI.  The ProgramDoc class also resides here.
 *
 * This file is part of MLPACK 1.0.9.
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
#ifndef __MLPACK_CORE_UTIL_OPTION_HPP
#define __MLPACK_CORE_UTIL_OPTION_HPP

#include <string>

#include "cli.hpp"

namespace mlpack {
namespace util {

/**
 * A static object whose constructor registers a parameter with the CLI class.
 * This should not be used outside of CLI itself, and you should use the
 * PARAM_FLAG(), PARAM_DOUBLE(), PARAM_INT(), PARAM_STRING(), or other similar
 * macros to declare these objects instead of declaring them directly.
 *
 * @see core/io/cli.hpp, mlpack::CLI
 */
template<typename N>
class Option
{
 public:
  /**
   * Construct an Option object.  When constructed, it will register
   * itself with CLI.
   *
   * @param ignoreTemplate Whether or not the template type matters for this
   *     option.  Essentially differs options with no value (flags) from those
   *     that do, and thus require a type.
   * @param defaultValue Default value this parameter will be initialized to.
   * @param identifier The name of the option (no dashes in front; for --help,
   *      we would pass "help").
   * @param description A short string describing the option.
   * @param parent Full pathname of the parent module that "owns" this option.
   *      The default is the root node (an empty string).
   * @param required Whether or not the option is required at runtime.
   */
  Option(bool ignoreTemplate,
         N defaultValue,
         const std::string& identifier,
         const std::string& description,
         const std::string& parent = std::string(""),
         bool required = false);

  /**
   * Constructs an Option object.  When constructed, it will register a flag
   * with CLI.
   *
   * @param identifier The name of the option (no dashes in front); for --help
   *     we would pass "help".
   * @param description A short string describing the option.
   * @param parent Full pathname of the parent module that "owns" this option.
   *     The default is the root node (an empty string).
   */
  Option(const std::string& identifier,
         const std::string& description,
         const std::string& parent = std::string(""));
};

/**
 * A static object whose constructor registers program documentation with the
 * CLI class.  This should not be used outside of CLI itself, and you should use
 * the PROGRAM_INFO() macro to declare these objects.  Only one ProgramDoc
 * object should ever exist.
 *
 * @see core/io/cli.hpp, mlpack::CLI
 */
class ProgramDoc
{
 public:
  /**
   * Construct a ProgramDoc object.  When constructed, it will register itself
   * with CLI.
   *
   * @param programName Short string representing the name of the program.
   * @param documentation Long string containing documentation on how to use the
   *     program and what it is.  No newline characters are necessary; this is
   *     taken care of by CLI later.
   */
  ProgramDoc(const std::string& programName,
             const std::string& documentation);

  //! The name of the program.
  std::string programName;
  //! Documentation for what the program does.
  std::string documentation;
};

}; // namespace util
}; // namespace mlpack

// For implementations of templated functions
#include "option_impl.hpp"

#endif
