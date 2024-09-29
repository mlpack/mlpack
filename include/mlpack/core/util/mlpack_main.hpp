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

#ifndef BINDING_NAME
  #error "BINDING_NAME not defined!"
#endif

#if (BINDING_TYPE == BINDING_TYPE_CLI) // This is a command-line executable.
  #include <mlpack/bindings/cli/mlpack_main.hpp>
#elif (BINDING_TYPE == BINDING_TYPE_TEST) // This is a unit test.
  #include <mlpack/bindings/tests/mlpack_main.hpp>
#elif (BINDING_TYPE == BINDING_TYPE_PYX) // This is a Python binding.
  #include <mlpack/bindings/python/mlpack_main.hpp>
#elif (BINDING_TYPE == BINDING_TYPE_JL) // This is a Julia binding.
  #include <mlpack/bindings/julia/mlpack_main.hpp>
#elif (BINDING_TYPE == BINDING_TYPE_GO) // This is a Go binding.
  #include <mlpack/bindings/go/mlpack_main.hpp>
#elif (BINDING_TYPE == BINDING_TYPE_R) // This is an R binding.
  #include <mlpack/bindings/R/mlpack_main.hpp>
#elif (BINDING_TYPE == BINDING_TYPE_MARKDOWN) // These are the Markdown docs.
  #include <mlpack/bindings/markdown/mlpack_main.hpp>
#else
  #error "Unknown binding type!  Be sure BINDING_TYPE is defined if you are " \
         "including <mlpack/core/util/mlpack_main.hpp>.";
#endif

#include "param_checks.hpp"

#endif
