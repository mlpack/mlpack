/**
 * @file cli_deleter.hpp
 * @author Ryan Curtin
 *
 * Definition of the CLIDeleter() class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_CLI_DELETER_HPP
#define MLPACK_CORE_UTIL_CLI_DELETER_HPP

namespace mlpack {
namespace util {

/**
 * Extremely simple class whose only job is to delete the existing CLI object at
 * the end of execution.  This is meant to allow the user to avoid typing
 * 'CLI::Destroy()' at the end of their program.  The file also defines a static
 * CLIDeleter class, which will be initialized at the beginning of the program
 * and deleted at the end.  The destructor destroys the CLI singleton.
 */
class CLIDeleter
{
 public:
  CLIDeleter();
  ~CLIDeleter();
};

} // namespace util
} // namespace mlpack

#endif
