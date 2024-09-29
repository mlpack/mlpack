/**
 * @file mlpack_main.hpp
 * @author Ryan Curtin
 *
 * Define the macros used when compiling a command-line binding.  This file
 * should not be included directly; instead, mlpack/core/util/mlpack_main.hpp
 * should be included with the right setting of BINDING_TYPE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_CLI_MLPACK_MAIN_HPP
#define MLPACK_BINDINGS_CLI_MLPACK_MAIN_HPP

#ifndef BINDING_TYPE
  #error "BINDING_TYPE not defined!  Don't include this file directly!"
#endif
#if BINDING_TYPE != BINDING_TYPE_CLI
  #error "BINDING_TYPE is not set to BINDING_TYPE_CLI!"
#endif

// Matrices are transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/cli/cli_option.hpp>
#include <mlpack/bindings/cli/print_doc_functions.hpp>

/**
 * PRINT_PARAM_STRING() returns a string that contains the correct
 * language-specific representation of a parameter's name.
 */
#define PRINT_PARAM_STRING(x) mlpack::bindings::cli::ParamString( \
    STRINGIFY(BINDING_NAME), x)

/**
 * PRINT_PARAM_VALUE() returns a string that contains a correct
 * language-specific representation of a parameter's value.
 */
#define PRINT_PARAM_VALUE mlpack::bindings::cli::PrintValue

/**
 * PRINT_DATASET() returns a string that contains a correct language-specific
 * representation of a dataset name.
 */
#define PRINT_DATASET mlpack::bindings::cli::PrintDataset

/**
 * PRINT_MODEL() returns a string that contains a correct language-specific
 * representation of an mlpack model name.
 */
#define PRINT_MODEL mlpack::bindings::cli::PrintModel

/**
 * PRINT_CALL() returns a string that contains the full language-specific
 * representation of a call to an mlpack binding.  The first argument should be
 * the name of the binding, and all other arguments should be names of
 * parameters followed by values (in the case where the preceding parameter is
 * not a flag).
 */
#define PRINT_CALL mlpack::bindings::cli::ProgramCall

/**
 * BINDING_IGNORE_CHECK() is an internally-used macro to determine whether or
 * not a specific parameter check should be ignored.
 */
#define BINDING_IGNORE_CHECK mlpack::bindings::cli::IgnoreCheck

/**
 * BINDING_MIN_LABEL is the minimum value a label can take, as represented in
 * the input binding language.  For CLI bindings, we expect the user to provide
 * their classes in the range [0, numClasses).
 */
#define BINDING_MIN_LABEL 0


namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::cli::CLIOption<T>;

}
}

#include <mlpack/core/util/param.hpp>
#include <mlpack/core/util/io.hpp>
#include <mlpack/bindings/cli/parse_command_line.hpp>
#include <mlpack/bindings/cli/end_program.hpp>

// Forward definition of the binding function.
void BINDING_FUNCTION(mlpack::util::Params&, mlpack::util::Timers&);

// Define the main function that will be used by this binding.
int main(int argc, char** argv)
{
  // Parse the command-line options; put them into CLI.
  mlpack::util::Params params =
      mlpack::bindings::cli::ParseCommandLine(argc, argv);
  // Create a new timer object for this call.
  mlpack::util::Timers timers;
  timers.Enabled() = true;
  mlpack::Timer::EnableTiming();

  // A "total_time" timer is run by default for each mlpack program.
  timers.Start("total_time");
  BINDING_FUNCTION(params, timers);
  timers.Stop("total_time");

  // Print output options, print verbose information, save model parameters,
  // clean up, and so forth.
  mlpack::bindings::cli::EndProgram(params, timers);
}

// Add default parameters that are included in every program.
PARAM_GLOBAL(bool, "help", "Default help info.", "h", "bool", false, true,
    false, false);
PARAM_GLOBAL(std::string, "info", "Print help on a specific option.", "",
    "std::string", false, true, false, "");
PARAM_GLOBAL(bool, "verbose", "Display informational messages and the full "
    "list of parameters and timers at the end of execution.", "v", "bool",
    false, true, false, false);
PARAM_GLOBAL(bool, "version", "Display the version of mlpack.", "V", "bool",
    false, true, false, false);

#endif
