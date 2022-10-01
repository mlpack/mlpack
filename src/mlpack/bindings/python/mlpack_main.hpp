/**
 * @file mlpack_main.hpp
 * @author Ryan Curtin
 *
 * Define the macros used when compiling a Python binding.  This file should not
 * be included directly; instead, mlpack/core/util/mlpack_main.hpp should be
 * included with the right setting of BINDING_TYPE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_PYTHON_MLPACK_MAIN_HPP
#define MLPACK_BINDINGS_PYTHON_MLPACK_MAIN_HPP

#ifndef BINDING_TYPE
  #error "BINDING_TYPE not defined!  Don't include this file directly!"
#endif
#if BINDING_TYPE != BINDING_TYPE_PYX
  #error "BINDING_TYPE is not set to BINDING_TYPE_PYX!"
#endif

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
 * IMPORT_EXT_LIB() returns a string that imports required external libraries
 * for a particular language.
 */
#define IMPORT_EXT_LIB mlpack::bindings::python::ImportExtLib

/**
 * IMPORT_SPLIT() returns a string that imports mlpack's preprocess_split.
 */
#define IMPORT_SPLIT mlpack::bindings::python::ImportSplit

/**
 * IMPORT_THIS returns a string that imports the current method.
 */
#define IMPORT_THIS mlpack::bindings::python::ImportThis

/**
 * GET_DATASET() returns a string that reads data from a source and,
 * stores in a variable.
 */
#define GET_DATASET mlpack::bindings::python::GetDataset

/**
 * SPLIT_TRAIN_TEST() splits the dataset into train and test datasets.
 */
#define SPLIT_TRAIN_TEST mlpack::bindings::python::SplitTrainTest

/**
 * CREATE_OBJECT() returns a string that creates an instance of the
 * class.
 */
#define CREATE_OBJECT(...) mlpack::bindings::python::CreateObject(\
    STRINGIFY(BINDING_NAME), __VA_ARGS__)

/**
 * CALL_METHOD() returns a string that calls a method of an instance.
 */
#define CALL_METHOD(...) mlpack::bindings::python::CallMethod(\
    STRINGIFY(BINDING_NAME), __VA_ARGS__)

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

/**
 * BINDING_MIN_LABEL is the minimum value a label can take, as represented in
 * the input binding language.  For Python bindings, we expect the user to
 * provide their classes in the range [0, numClasses).
 */
#define BINDING_MIN_LABEL 0

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

// Define parameters available in every Python binding.
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

#endif
