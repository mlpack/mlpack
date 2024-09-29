/**
 * @file test_function_map.hpp
 * @author Ryan Curtin
 *
 * Define a singleton that will hold the functionMap for any TestOptions.  This
 * is necessary because the same IO instance may deal with multiple types of
 * options in `mlpack_test`, and so we want to specifically register all
 * functions associated with a TestOption so that they can be easily restored
 * during a test.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_BINDINGS_TESTS_TEST_FUNCTION_MAP_HPP
#define MLPACK_BINDINGS_TESTS_TEST_FUNCTION_MAP_HPP

#include <mlpack/core/util/param_data.hpp>

namespace mlpack {
namespace bindings {
namespace tests {

class TestFunctionMap
{
 public:
  // Convenience typedef.
  typedef std::map<std::string, std::map<std::string,
      void (*)(util::ParamData&, const void*, void*)>> FunctionMapType;

  //! Get the instantiated TestFunctionMap object.
  static TestFunctionMap& GetSingleton();

  /**
   * Register the function `func` for the given typename `tname` and the given
   * function name `functionName`.
   */
  static void RegisterFunction(
      const std::string& tname,
      const std::string& functionName,
      void (*func)(util::ParamData&, const void*, void*));

  //! Get the populated function map.
  static const FunctionMapType& FunctionMap();

 private:
  //! The private constructor means that extra instances are not allowed.
  TestFunctionMap();

  //! Map of functions.  Note that this is not specific to a binding, so we only
  //! have one.
  FunctionMapType functionMap;
};

} // namespace tests
} // namespace bindings
} // namespace mlpack

#endif
