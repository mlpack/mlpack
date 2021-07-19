/**
 * @param mlpack_main.hpp
 * @author Ryan Curtin
 *
 * This file, based on the value of the macro BINDING_TYPE, will define the
 * macros necessary to compile an mlpack binding for the target language.
 *
 * This file should *only* be included by a program that is meant to be a
 * command-line program or a binding to another language.  This file also
 * includes param_checks.hpp, which contains functions that are used to check
 * parameter values at runtime.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_CORE_UTIL_MLPACK_MAIN_HPP
#define MLPACK_CORE_UTIL_MLPACK_MAIN_HPP

#define BINDING_TYPE_CLI 0
#define BINDING_TYPE_TEST 1
#define BINDING_TYPE_PYX 2
#define BINDING_TYPE_JL 3
#define BINDING_TYPE_GO 4
#define BINDING_TYPE_R 5
#define BINDING_TYPE_MARKDOWN 128
#define BINDING_TYPE_UNKNOWN -1

#ifndef BINDING_TYPE
#define BINDING_TYPE BINDING_TYPE_UNKNOWN
#endif

#if (BINDING_TYPE == BINDING_TYPE_CLI) // This is a command-line executable.

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
 * PRINT_CALL() returns a string that contains the full language-specific
 * representation of a call to an mlpack binding.  The first argument should be
 * the name of the binding, and all other arguments should be names of
 * parameters followed by values (in the case where the preceding parameter is
 * not a flag).
 */
#define PRINT_CALL mlpack::bindings::cli::ProgramCall

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
 * BINDING_IGNORE_CHECK() is an internally-used macro to determine whether or
 * not a specific parameter check should be ignored.
 */
#define BINDING_IGNORE_CHECK mlpack::bindings::cli::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::cli::CLIOption<T>;

}
}

#include <mlpack/core/util/param.hpp>
#include <mlpack/bindings/cli/parse_command_line.hpp>
#include <mlpack/bindings/cli/end_program.hpp>

#ifndef BINDING_NAME
  #error "BINDING_NAME not defined!"
#endif

void BINDING_NAME(mlpack::util::Params&, mlpack::util::Timers&);

int main(int argc, char** argv)
{
  // Parse the command-line options; put them into CLI.
  mlpack::util::Params params =
      mlpack::bindings::cli::ParseCommandLine(argc, argv);
  // Create a new timer object for this call.
  mlpack::util::Timers timers;
  timers.Enabled() = true;

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

#elif (BINDING_TYPE == BINDING_TYPE_TEST) // This is a unit test.

// Matrices are not transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED false

#include <mlpack/bindings/tests/test_option.hpp>
#include <mlpack/bindings/tests/ignore_check.hpp>
#include <mlpack/bindings/tests/clean_memory.hpp>

// These functions will do nothing.
#define PRINT_PARAM_STRING(A) std::string(" ")
#define PRINT_PARAM_VALUE(A, B) std::string(" ")
#define PRINT_DATASET(A) std::string(" ")
#define PRINT_MODEL(A) std::string(" ")

/**
 * PRINT_CALL() returns a string that contains the full language-specific
 * representation of a call to an mlpack binding.  The first argument should be
 * the name of the binding, and all other arguments should be names of
 * parameters followed by values (in the case where the preceding parameter is
 * not a flag).
 */
#define PRINT_CALL(...) std::string(" ")

/**
 * BINDING_IGNORE_CHECK() is an internally-used macro to determine whether or
 * not a specific parameter check should be ignored.
 */
#define BINDING_IGNORE_CHECK mlpack::bindings::tests::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::tests::TestOption<T>;

}
}

#include <mlpack/core/util/param.hpp>

// For the tests, we want to call the binding function
// mlpack_test_<BINDING_NAME>() instead of just <BINDING_NAME>(), so we change
// the definition of BINDING_FUNCTION().  This is to avoid namespace/function
// ambiguities.
#undef BINDING_FUNCTION
#define BINDING_FUNCTION(...) JOIN(mlpack_test_, BINDING_NAME)(__VA_ARGS__)

#elif(BINDING_TYPE == BINDING_TYPE_PYX) // This is a Python binding.

// Matrices are transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/python/py_option.hpp>
#include <mlpack/bindings/python/print_doc_functions.hpp>

/**
 * PRINT_PARAM_STRING() returns a string that contains the correct
 * language-specific representation of a parameter's name.
 */
#define PRINT_PARAM_STRING mlpack::bindings::python::ParamString

/**
 * PRINT_PARAM_VALUE() returns a string that contains a correct
 * language-specific representation of a parameter's value.
 */
#define PRINT_PARAM_VALUE mlpack::bindings::python::PrintValue

/**
 * PRINT_DATASET() returns a string that contains a correct language-specific
 * representation of a dataset name.
 */
#define PRINT_DATASET mlpack::bindings::python::PrintDataset

/**
 * PRINT_MODEL() returns a string that contains a correct language-specific
 * representation of an mlpack model name.
 */
#define PRINT_MODEL mlpack::bindings::python::PrintModel

/**
 * PRINT_CALL() returns a string that contains the full language-specific
 * representation of a call to an mlpack binding.  The first argument should be
 * the name of the binding, and all other arguments should be names of
 * parameters followed by values (in the case where the preceding parameter is
 * not a flag).
 */
#define PRINT_CALL mlpack::bindings::python::ProgramCall

/**
 * BINDING_IGNORE_CHECK() is an internally-used macro to determine whether or
 * not a specific parameter check should be ignored.
 */
#define BINDING_IGNORE_CHECK(x) mlpack::bindings::python::IgnoreCheck( \
    STRINGIFY(BINDING_NAME), x)

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::python::PyOption<T>;

}
}

#include <mlpack/core/util/param.hpp>

// In Python, we want to call the binding function mlpack_<BINDING_NAME>()
// instead of just <BINDING_NAME>(), so we change the definition of
// BINDING_FUNCTION().
#undef BINDING_FUNCTION
#define BINDING_FUNCTION(...) JOIN(mlpack_, BINDING_NAME)(__VA_ARGS__)

#ifndef BINDING_NAME
  #error "BINDING_NAME not defined!"
#endif

PARAM_GLOBAL(bool, "verbose", "Display informational messages and the full "
    "list of parameters and timers at the end of execution.", "v", "bool",
    false, true, false, false);
PARAM_GLOBAL(bool, "copy_all_inputs", "If specified, all input parameters "
    "will be deep copied before the method is run.  This is useful for "
    "debugging problems where the input parameters are being modified "
    "by the algorithm, but can slow down the code.", "", "bool",
    false, true, false, false);
PARAM_GLOBAL(bool, "check_input_matrices", "If specified, the input matrix "
    "is checked for NaN and inf values; an exception is thrown if any are "
    "found.", "", "bool", false, true, false, false);

// Nothing else needs to be defined---the binding will use BINDING_NAME() as-is.

#elif(BINDING_TYPE == BINDING_TYPE_JL) // This is a Julia binding.

// Matrices are transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/julia/julia_option.hpp>
#include <mlpack/bindings/julia/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::julia::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::julia::PrintValue
#define PRINT_DATASET mlpack::bindings::julia::PrintDataset
#define PRINT_MODEL mlpack::bindings::julia::PrintModel
#define PRINT_CALL(...) mlpack::bindings::julia::ProgramCall( \
    STRINGIFY(BINDING_NAME), __VA_ARGS__)
#define BINDING_IGNORE_CHECK(...) mlpack::bindings::julia::IgnoreCheck( \
    STRINGIFY(BINDING_NAME), __VA_ARGS__)

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::julia::JuliaOption<T>;

}
}

#include <mlpack/core/util/param.hpp>

PARAM_GLOBAL(bool, "verbose", "Display informational messages and the full "
    "list of parameters and timers at the end of execution.", "v", "bool",
    false, true, false, false);

// Nothing else needs to be defined---the binding will use mlpackMain() as-is.

#elif(BINDING_TYPE == BINDING_TYPE_GO) // This is a Go binding.

// Matrices are transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/go/go_option.hpp>
#include <mlpack/bindings/go/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::go::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::go::PrintValue
#define PRINT_DATASET mlpack::bindings::go::PrintDataset
#define PRINT_MODEL mlpack::bindings::go::PrintModel
#define PRINT_CALL mlpack::bindings::go::ProgramCall
#define BINDING_IGNORE_CHECK(x) mlpack::bindings::go::IgnoreCheck( \
    STRINGIFY(BINDING_NAME), x)

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::go::GoOption<T>;

}
}

#include <mlpack/core/util/param.hpp>

// In Go, we want to call the binding function mlpack_<BINDING_NAME>() instead
// of just <BINDING_NAME>(), so we change the definition of BINDING_FUNCTION().
#undef BINDING_FUNCTION
#define BINDING_FUNCTION(...) JOIN(mlpack_, BINDING_NAME)(__VA_ARGS__)

PARAM_GLOBAL(bool, "verbose", "Display informational messages and the full "
    "list of parameters and timers at the end of execution.", "v", "bool",
    false, true, false, false);

// Nothing else needs to be defined---the binding will use mlpackMain() as-is.

#elif(BINDING_TYPE == BINDING_TYPE_R) // This is a R binding.

// This doesn't actually matter for this binding type.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/R/R_option.hpp>
#include <mlpack/bindings/R/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::r::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::r::PrintValue
#define PRINT_DATASET mlpack::bindings::r::PrintDataset
#define PRINT_MODEL mlpack::bindings::r::PrintModel
#define PRINT_CALL(...) mlpack::bindings::r::ProgramCall(false, __VA_ARGS__)
#define BINDING_IGNORE_CHECK(...) mlpack::bindings::r::IgnoreCheck( \
    STRINGIFY(BINDING_NAME), __VA_ARGS__)

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::r::ROption<T>;

}
}

#include <mlpack/core/util/param.hpp>

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");

#undef BINDING_FUNCTION
#define BINDING_FUNCTION(...) JOIN(mlpack_, BINDING_NAME)(__VA_ARGS__)

// Nothing else needs to be defined---the binding will use mlpackMain() as-is.

#elif BINDING_TYPE == BINDING_TYPE_MARKDOWN

// This value doesn't actually matter, but it needs to be defined as something.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/markdown/md_option.hpp>
#include <mlpack/bindings/markdown/print_doc_functions.hpp>

/**
 * PRINT_PARAM_STRING() returns a string that contains the correct
 * language-specific representation of a parameter's name.
 */
#define PRINT_PARAM_STRING(x) mlpack::bindings::markdown::ParamString( \
    STRINGIFY(BINDING_NAME), x)

/**
 * PRINT_PARAM_VALUE() returns a string that contains a correct
 * language-specific representation of a parameter's value.
 */
#define PRINT_PARAM_VALUE mlpack::bindings::markdown::PrintValue

/**
 * PRINT_DATASET() returns a string that contains a correct language-specific
 * representation of a dataset name.
 */
#define PRINT_DATASET mlpack::bindings::markdown::PrintDataset

/**
 * PRINT_MODEL() returns a string that contains a correct language-specific
 * representation of an mlpack model name.
 */
#define PRINT_MODEL mlpack::bindings::markdown::PrintModel

/**
 * PRINT_CALL() returns a string that contains the full language-specific
 * representation of a call to an mlpack binding.  The first argument should be
 * the name of the binding, and all other arguments should be names of
 * parameters followed by values (in the case where the preceding parameter is
 * not a flag).
 */
#define PRINT_CALL mlpack::bindings::markdown::ProgramCall

/**
 * BINDING_IGNORE_CHECK() is an internally-used macro to determine whether or
 * not a specific parameter check should be ignored.
 */
#define BINDING_IGNORE_CHECK(x) mlpack::bindings::markdown::IgnoreCheck( \
    STRINGIFY(BINDING_NAME), x)

// This doesn't actually matter for this binding type.
#define BINDING_MATRIX_TRANSPOSED true

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::markdown::MDOption<T>;

}
}

#include <mlpack/core/util/param.hpp>

/*
#include <mlpack/bindings/markdown/program_doc_wrapper.hpp>

#undef BINDING_USER_NAME
#undef BINDING_SHORT_DESC
#undef BINDING_LONG_DESC
#undef BINDING_EXAMPLE
#undef BINDING_SEE_ALSO

#define BINDING_USER_NAME(NAME) static \
    mlpack::bindings::markdown::ProgramNameWrapper \
    io_programname_dummy_object = \
    mlpack::bindings::markdown::ProgramNameWrapper( \
    STRINGIFY(BINDING_NAME), NAME);

#define BINDING_SHORT_DESC(SHORT_DESC) static \
    mlpack::bindings::markdown::ShortDescriptionWrapper \
    io_programshort_desc_dummy_object = \
    mlpack::bindings::markdown::ShortDescriptionWrapper( \
    STRINGIFY(BINDING_NAME), SHORT_DESC);

#define BINDING_LONG_DESC(LONG_DESC) static \
    mlpack::bindings::markdown::LongDescriptionWrapper \
    io_programlong_desc_dummy_object = \
    mlpack::bindings::markdown::LongDescriptionWrapper( \
    STRINGIFY(BINDING_NAME), []() { return std::string(LONG_DESC); });

#ifdef __COUNTER__
  #define BINDING_EXAMPLE(EXAMPLE) static \
      mlpack::bindings::markdown::ExampleWrapper \
      JOIN(io_programexample_dummy_object_, __COUNTER__) = \
      mlpack::bindings::markdown::ExampleWrapper(STRINGIFY(BINDING_NAME), \
      []() { return(std::string(EXAMPLE)); });

  #define BINDING_SEE_ALSO(DESCRIPTION, LINK) static \
      mlpack::bindings::markdown::SeeAlsoWrapper \
      JOIN(io_programsee_also_dummy_object_, __COUNTER__) = \
      mlpack::bindings::markdown::SeeAlsoWrapper(STRINGIFY(BINDING_NAME), \
      DESCRIPTION, LINK);
#else
  #define BINDING_EXAMPLE(EXAMPLE) static \
      mlpack::bindings::markdown::ExampleWrapper \
      JOIN(JOIN(io_programexample_dummy_object_, __LINE__), opt) = \
      mlpack::bindings::markdown::ExampleWrapper(STRINGIFY(BINDING_NAME), \
      []() { return(std::string(EXAMPLE)); });

  #define BINDING_SEE_ALSO(DESCRIPTION, LINK) static \
      mlpack::bindings::markdown::SeeAlsoWrapper \
      JOIN(JOIN(io_programsee_also_dummy_object_, __LINE__), opt) = \
      mlpack::bindings::markdown::SeeAlsoWrapper(STRINGIFY(BINDING_NAME), \
      DESCRIPTION, LINK);
#endif
*/

// This parameter is available for all languages.
PARAM_GLOBAL(bool, "verbose", "Display informational messages and the full "
    "list of parameters and timers at the end of execution.", "v", "bool",
    false, true, false, false);

// CLI-specific parameters.
PARAM_GLOBAL(bool, "help", "Default help info.", "h", "bool", false, true,
    false, false);
PARAM_GLOBAL(std::string, "info", "Print help on a specific option.", "",
    "std::string", false, true, false, "");
PARAM_GLOBAL(bool, "version", "Display the version of mlpack.", "V", "bool",
    false, true, false, false);

// Python-specific parameters.
PARAM_GLOBAL(bool, "copy_all_inputs", "If specified, all input parameters will be"
    " deep copied before the method is run.  This is useful for debugging "
    "problems where the input parameters are being modified by the algorithm, but"
    " can slow down the code.", "", "bool", false, true, false, false);
PARAM_GLOBAL(bool, "check_input_matrices", "If specified, the input matrix "
    "is checked for NaN and inf values; an exception is thrown if any are "
    "found.", "", "bool", false, true, false, false);

#else

#error "Unknown binding type!  Be sure BINDING_TYPE is defined if you are " \
       "including <mlpack/core/util/mlpack_main.hpp>.";

#endif

#include "param_checks.hpp"

#endif
