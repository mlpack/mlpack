/**
 * @file print_help.hpp
 * @author Ryan Curtin
 *
 * Print help for a command-line program.
 */
#ifndef MLPACK_BINDINGS_CLI_PRINT_HELP_HPP
#define MLPACK_BINDINGS_CLI_PRINT_HELP_HPP

#include <mlpack/core.hpp>

namespace mlpack {
namespace bindings {
namespace cli {

/**
 * Print the help for the given parameter.  If no parameter is specified, then
 * help will be printed for all parameters.
 *
 * @param param Parameter name to print help for.
 */
void PrintHelp(const std::string& param = "");

} // namespace cli
} // namespace bindings
} // namespace mlpack

#endif
