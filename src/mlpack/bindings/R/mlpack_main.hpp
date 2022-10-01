/**
 * @file mlpack_main.hpp
 * @author Ryan Curtin
 *
 * Define the macros used when compiling a Go binding.  This file should not
 * be included directly; instead, mlpack/core/util/mlpack_main.hpp should be
 * included with the right setting of BINDING_TYPE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_R_MLPACK_MAIN_HPP
#define MLPACK_BINDINGS_R_MLPACK_MAIN_HPP

#ifndef BINDING_TYPE
  #error "BINDING_TYPE not defined!  Don't include this file directly!"
#endif
#if BINDING_TYPE != BINDING_TYPE_R
  #error "BINDING_TYPE is not set to BINDING_TYPE_R!"
#endif

// This doesn't actually matter for this binding type.
#define BINDING_MATRIX_TRANSPOSED true

#include <mlpack/bindings/R/R_option.hpp>
#include <mlpack/bindings/R/print_doc_functions.hpp>

/**
 * PRINT_PARAM_STRING() returns a string that contains the correct
 * language-specific representation of a parameter's name.
 */
#define PRINT_PARAM_STRING mlpack::bindings::r::ParamString

/**
 * PRINT_PARAM_VALUE() returns a string that contains a correct
 * language-specific representation of a parameter's value.
 */
#define PRINT_PARAM_VALUE mlpack::bindings::r::PrintValue

/**
 * PRINT_DATASET() returns a string that contains a correct language-specific
 * representation of a dataset name.
 */
#define PRINT_DATASET mlpack::bindings::r::PrintDataset

/**
 * PRINT_MODEL() returns a string that contains a correct language-specific
 * representation of an mlpack model name.
 */
#define PRINT_MODEL mlpack::bindings::r::PrintModel

/**
 * PRINT_CALL() returns a string that contains the full language-specific
 * representation of a call to an mlpack binding.  The first argument should be
 * the name of the binding, and all other arguments should be names of
 * parameters followed by values (in the case where the preceding parameter is
 * not a flag).
 */
#define PRINT_CALL(...) mlpack::bindings::r::ProgramCall(false, __VA_ARGS__)

/**
 * BINDING_IGNORE_CHECK() is an internally-used macro to determine whether or
 * not a specific parameter check should be ignored.
 */
#define BINDING_IGNORE_CHECK(...) mlpack::bindings::r::IgnoreCheck( \
    STRINGIFY(BINDING_NAME), __VA_ARGS__)

/**
 * BINDING_MIN_LABEL is the minimum value a label can take, as represented in
 * the input binding language.  For R bindings, we expect the user to provide
 * their classes in the range [0, numClasses).
 */
#define BINDING_MIN_LABEL 0

namespace mlpack {
namespace util {

template<typename T>
using Option = mlpack::bindings::r::ROption<T>;

}
}

#include <mlpack/core/util/param.hpp>

// In R, we want to call the binding function mlpack_<BINDING_NAME>() instead of
// just <BINDING_NAME>(), so we change the definition of BINDING_FUNCTION().
#undef BINDING_FUNCTION
#define BINDING_FUNCTION(...) JOIN(mlpack_, BINDING_NAME)(__VA_ARGS__)

// Add default parameters that are included in every program.
PARAM_FLAG("verbose", "Display informational messages and the full list of "
    "parameters and timers at the end of execution.", "v");

#endif
