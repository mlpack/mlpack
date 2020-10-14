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
#define PRINT_PARAM_STRING mlpack::bindings::cli::ParamString

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

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>
#include <mlpack/bindings/cli/parse_command_line.hpp>
#include <mlpack/bindings/cli/end_program.hpp>

static void mlpackMain(); // This is typically defined after this include.

int main(int argc, char** argv)
{
  // Parse the command-line options; put them into CLI.
  mlpack::bindings::cli::ParseCommandLine(argc, argv);
  // Enable timing.
  mlpack::Timer::EnableTiming();

  // A "total_time" timer is run by default for each mlpack program.
  mlpack::Timer::Start("total_time");

  mlpackMain();

  // Print output options, print verbose information, save model parameters,
  // clean up, and so forth.
  mlpack::bindings::cli::EndProgram();
}

#elif(BINDING_TYPE == BINDING_TYPE_TEST) // This is a unit test.

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

// testName symbol should be defined in each binding test file
#include <mlpack/core/util/param.hpp>

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
#define BINDING_IGNORE_CHECK mlpack::bindings::python::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::python::PyOption<T>;

}
}

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>

#undef BINDING_NAME
#define BINDING_NAME(NAME) static \
    mlpack::util::ProgramName \
    io_programname_dummy_object = mlpack::util::ProgramName(NAME); \
    namespace mlpack { \
    namespace bindings { \
    namespace python { \
    std::string programName = NAME; \
    } \
    } \
    }

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");
PARAM_FLAG("copy_all_inputs", "If specified, all input parameters will be deep"
    " copied before the method is run.  This is useful for debugging problems "
    "where the input parameters are being modified by the algorithm, but can "
    "slow down the code.", "");

// Nothing else needs to be defined---the binding will use mlpackMain() as-is.

#elif(BINDING_TYPE == BINDING_TYPE_JL) // This is a Julia binding.

// Matrices are transposed on load/save.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/julia/julia_option.hpp>
#include <mlpack/bindings/julia/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::julia::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::julia::PrintValue
#define PRINT_DATASET mlpack::bindings::julia::PrintDataset
#define PRINT_MODEL mlpack::bindings::julia::PrintModel
#define PRINT_CALL mlpack::bindings::julia::ProgramCall
#define BINDING_IGNORE_CHECK mlpack::bindings::julia::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::julia::JuliaOption<T>;

}
}

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>

#undef BINDING_NAME
#define BINDING_NAME(NAME) static \
    mlpack::util::ProgramName \
    io_programname_dummy_object = mlpack::util::ProgramName(NAME); \
    namespace mlpack { \
    namespace bindings { \
    namespace julia { \
    std::string programName = NAME; \
    } \
    } \
    }

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");

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
#define BINDING_IGNORE_CHECK mlpack::bindings::go::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::go::GoOption<T>;

}
}

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>

#undef BINDING_NAME
#define BINDING_NAME(NAME) static \
    mlpack::util::ProgramName \
    io_programname_dummy_object = mlpack::util::ProgramName(NAME); \
    namespace mlpack { \
    namespace bindings { \
    namespace go { \
    std::string programName = NAME; \
    } \
    } \
    }

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");

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
#define BINDING_IGNORE_CHECK mlpack::bindings::r::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::r::ROption<T>;

}
}

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");

// Nothing else needs to be defined---the binding will use mlpackMain() as-is.

#elif BINDING_TYPE == BINDING_TYPE_MARKDOWN

// We use MARKDOWN_BINDING_NAME in BINDING_NAME(), BINDING_SHORT_DESC(),
// BINDING_LONG_DESC(), BINDING_EXAMPLE() and BINDING_SEE_ALSO()
// so it needs to be defined.
#ifndef MARKDOWN_BINDING_NAME
  #error "MARKDOWN_BINDING_NAME must be defined when BINDING_TYPE is Markdown!"
#endif

// This value doesn't actually matter, but it needs to be defined as something.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/markdown/md_option.hpp>
#include <mlpack/bindings/markdown/print_doc_functions.hpp>

/**
 * PRINT_PARAM_STRING() returns a string that contains the correct
 * language-specific representation of a parameter's name.
 */
#define PRINT_PARAM_STRING mlpack::bindings::markdown::ParamString

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
#define BINDING_IGNORE_CHECK mlpack::bindings::markdown::IgnoreCheck

// This doesn't actually matter for this binding type.
#define BINDING_MATRIX_TRANSPOSED true

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::markdown::MDOption<T>;

}
}

#include <mlpack/core/util/param.hpp>
#include <mlpack/bindings/markdown/program_doc_wrapper.hpp>

#undef BINDING_NAME
#undef BINDING_SHORT_DESC
#undef BINDING_LONG_DESC
#undef BINDING_EXAMPLE
#undef BINDING_SEE_ALSO

#define BINDING_NAME(NAME) static \
    mlpack::bindings::markdown::ProgramNameWrapper \
    io_programname_dummy_object = \
    mlpack::bindings::markdown::ProgramNameWrapper( \
    MARKDOWN_BINDING_NAME, NAME);

#define BINDING_SHORT_DESC(SHORT_DESC) static \
    mlpack::bindings::markdown::ShortDescriptionWrapper \
    io_programshort_desc_dummy_object = \
    mlpack::bindings::markdown::ShortDescriptionWrapper( \
    MARKDOWN_BINDING_NAME, SHORT_DESC);

#define BINDING_LONG_DESC(LONG_DESC) static \
    mlpack::bindings::markdown::LongDescriptionWrapper \
    io_programlong_desc_dummy_object = \
    mlpack::bindings::markdown::LongDescriptionWrapper( \
    MARKDOWN_BINDING_NAME, []() { return std::string(LONG_DESC); });

#ifdef __COUNTER__
  #define BINDING_EXAMPLE(EXAMPLE) static \
      mlpack::bindings::markdown::ExampleWrapper \
      JOIN(io_programexample_dummy_object_, __COUNTER__) = \
      mlpack::bindings::markdown::ExampleWrapper(MARKDOWN_BINDING_NAME, \
      []() { return(std::string(EXAMPLE)); });

  #define BINDING_SEE_ALSO(DESCRIPTION, LINK) static \
      mlpack::bindings::markdown::SeeAlsoWrapper \
      JOIN(io_programsee_also_dummy_object_, __COUNTER__) = \
      mlpack::bindings::markdown::SeeAlsoWrapper(MARKDOWN_BINDING_NAME, \
      DESCRIPTION, LINK);
#else
  #define BINDING_EXAMPLE(EXAMPLE) static \
      mlpack::bindings::markdown::ExampleWrapper \
      JOIN(JOIN(io_programexample_dummy_object_, __LINE__), opt) = \
      mlpack::bindings::markdown::ExampleWrapper(MARKDOWN_BINDING_NAME, \
      []() { return(std::string(EXAMPLE)); });

  #define BINDING_SEE_ALSO(DESCRIPTION, LINK) static \
      mlpack::bindings::markdown::SeeAlsoWrapper \
      JOIN(JOIN(io_programsee_also_dummy_object_, __LINE__), opt) = \
      mlpack::bindings::markdown::SeeAlsoWrapper(MARKDOWN_BINDING_NAME, \
      DESCRIPTION, LINK);
#endif

PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");

// CLI-specific parameters.
PARAM_FLAG("help", "Default help info.", "h");
PARAM_STRING_IN("info", "Print help on a specific option.", "", "");
PARAM_FLAG("version", "Display the version of mlpack.", "V");

// Python-specific parameters.
PARAM_FLAG("copy_all_inputs", "If specified, all input parameters will be deep"
    " copied before the method is run.  This is useful for debugging problems "
    "where the input parameters are being modified by the algorithm, but can "
    "slow down the code.", "");

#else

#error "Unknown binding type!  Be sure BINDING_TYPE is defined if you are " \
       "including <mlpack/core/util/mlpack_main.hpp>.";

#endif

#include "param_checks.hpp"

#endif
