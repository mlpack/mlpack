/**
 * @file mlpack_main.hpp
 * @author Ryan Curtin
 *
 * Define the macros used when compiling a test binding.  This file should not
 * be included directly; instead, mlpack/core/util/mlpack_main.hpp should be
 * included with the right setting of BINDING_TYPE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_TESTS_MLPACK_MAIN_HPP
#define MLPACK_BINDINGS_TESTS_MLPACK_MAIN_HPP

#ifndef BINDING_TYPE
  #error "BINDING_TYPE not defined!  Don't include this file directly!"
#endif
#if BINDING_TYPE != BINDING_TYPE_TEST
  #error "BINDING_TYPE is not set to BINDING_TYPE_TEST!"
#endif

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
#define PRINT_CALL(...) std::string(" ")
#define BINDING_IGNORE_CHECK mlpack::bindings::tests::IgnoreCheck
#define IMPORT_EXT_LIB(...) std::string(" ")
#define IMPORT_SPLIT(...) std::string(" ")
#define IMPORT_THIS(...) std::string(" ")
#define GET_DATASET(...) std::string(" ")
#define SPLIT_TRAIN_TEST(...) std::string(" ")
#define CREATE_OBJECT(...) std::string(" ")
#define CALL_METHOD(...) std::string(" ")

// This doesn't actually matter for this binding type.
#define BINDING_MIN_LABEL 0

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

#endif
