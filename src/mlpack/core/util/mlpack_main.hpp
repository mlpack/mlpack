/**
 * @param mlpack_cli_main.hpp
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
#define BINDING_TYPE_UNKNOWN -1

#ifndef BINDING_TYPE
#define BINDING_TYPE BINDING_TYPE_UNKNOWN
#endif

#if (BINDING_TYPE == BINDING_TYPE_CLI) // This is a command-line executable.

#include <mlpack/bindings/cli/cli_option.hpp>
#include <mlpack/bindings/cli/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::cli::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::cli::PrintValue
#define PRINT_CALL mlpack::bindings::cli::ProgramCall
#define PRINT_DATASET mlpack::bindings::cli::PrintDataset
#define PRINT_MODEL mlpack::bindings::cli::PrintModel
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

#include <mlpack/bindings/tests/test_option.hpp>
#include <mlpack/bindings/tests/ignore_check.hpp>
#include <mlpack/bindings/tests/clean_memory.hpp>

// These functions will do nothing.
#define PRINT_PARAM_STRING(A) std::string(" ")
#define PRINT_PARAM_VALUE(A, B) std::string(" ")
#define PRINT_DATASET(A) std::string(" ")
#define PRINT_MODEL(A) std::string(" ")
#define PRINT_CALL(...) std::string(" ")
#define BINDING_IGNORE_CHECK mlpack::bindings::tests::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::tests::TestOption<T>;

}
}

// testName symbol should be defined in each binding test file
#include <mlpack/core/util/param.hpp>

#undef PROGRAM_INFO
#define PROGRAM_INFO(NAME, DESC) static mlpack::util::ProgramDoc \
    cli_programdoc_dummy_object = mlpack::util::ProgramDoc(NAME, \
        []() { return DESC; });

#elif(BINDING_TYPE == BINDING_TYPE_PYX) // This is a Python binding.

#include <mlpack/bindings/python/py_option.hpp>
#include <mlpack/bindings/python/print_doc_functions.hpp>

#define PRINT_PARAM_STRING mlpack::bindings::python::ParamString
#define PRINT_PARAM_VALUE mlpack::bindings::python::PrintValue
#define PRINT_DATASET mlpack::bindings::python::PrintDataset
#define PRINT_MODEL mlpack::bindings::python::PrintModel
#define PRINT_CALL mlpack::bindings::python::ProgramCall
#define BINDING_IGNORE_CHECK mlpack::bindings::python::IgnoreCheck

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::python::PyOption<T>;

}
}

static const std::string testName = "";
#include <mlpack/core/util/param.hpp>

#undef PROGRAM_INFO
#define PROGRAM_INFO(NAME, DESC) static mlpack::util::ProgramDoc \
    cli_programdoc_dummy_object = mlpack::util::ProgramDoc(NAME, \
        []() { return DESC; }); \
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

#else

#error "Unknown binding type!  Be sure BINDING_TYPE is defined if you are " \
       "including <mlpack/core/util/mlpack_main.hpp>.";

#endif

#include "param_checks.hpp"

#endif
