/**
 * @file option.hpp
 * @author Matthew Amidon
 *
 * Definition of the Option class, which is used to define parameters which are
 * used by CLI.  The ProgramDoc class also resides here.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_OPTION_HPP
#define MLPACK_CORE_UTIL_OPTION_HPP

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
   * @param defaultValue Default value this parameter will be initialized to
   *      (for flags, this should be false, for instance).
   * @param identifier The name of the option (no dashes in front; for --help,
   *      we would pass "help").
   * @param description A short string describing the option.
   * @param alias Short name of the parameter. "" for no alias.
   * @param cppName Name of the C++ type of this parameter (i.e. "int").
   * @param required Whether or not the option is required at runtime.
   * @param input Whether or not the option is an input option.
   * @param noTranspose If the parameter is a matrix and this is true, then the
   *      matrix will not be transposed on loading.
   */
  Option(const N defaultValue,
         const std::string& identifier,
         const std::string& description,
         const std::string& alias,
         const std::string& cppName,
         const bool required = false,
         const bool input = true,
         const bool noTranspose = false)
  {
    // Do nothing!
  }
};

} // namespace util
} // namespace mlpack

#endif
