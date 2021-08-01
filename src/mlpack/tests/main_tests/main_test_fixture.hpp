/**
 * @file tests/main_tests/main_test_fixture.hpp
 * @author Ryan Curtin
 *
 * Implementation of MainTestFixture, the base class for the test fixture for
 * all main classes.  This also defines the MAIN_TEST_FIXTURE() convenience
 * macro.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MLPACK_TESTS_MAIN_TEST_FIXTURE_HPP
#define MLPACK_TESTS_MAIN_TEST_FIXTURE_HPP

#include <mlpack/core.hpp>
#include <mlpack/core/util/params.hpp>
#include <mlpack/bindings/tests/test_function_map.hpp>

/**
 * Define a test fixture for an mlpack binding test with the name given as
 * `CLASS_NAME`.  This fixture has all the same methods as `MainTestFixture` and
 * can be used to set input parameters for a binding test.  When all the
 * parameters are set, use the `RUN_BINDING()` macro to actually run the
 * binding.
 *
 * Before calling this macro, make sure the `BINDING_NAME` macro is defined
 * appropriately.  This is generally done simply by including the binding's
 * `*_main.cpp` file.
 */
#define BINDING_TEST_FIXTURE(CLASS_NAME) \
    class CLASS_NAME : public MainTestFixture \
    { \
     public: \
      CLASS_NAME() : MainTestFixture(IO::Parameters(STRINGIFY(BINDING_NAME))) \
      { } \
    };

/**
 * Run the binding.  This depends on the `BINDING_NAME` macro being defined
 * appropriately.  This is generally done simply by including the binding's
 * `*_main.cpp` file.
 */
#define RUN_BINDING() BINDING_FUNCTION(params, timers)

/**
 * MainTestFixture is a base class for Catch fixtures for mlpack binding tests.
 * Instead of using this class directly, use the `BINDING_TEST_FIXTURE()` macro
 * to correctly define a fixture once `BINDING_NAME` is defined in your test
 * file.  Then, you can use all the methods in this class inside the tests.
 */
class MainTestFixture
{
 public:
  //! Create a MainTestFixture with the given set of parameters.
  MainTestFixture(const util::Params& paramsIn) :
      paramsClean(paramsIn)
  {
    // For a test binding, the parameters will not have the correct maps in
    // `functionMap`.  This is because the test executable may also be adding
    // options for different binding types (which happens in
    // `cli_binding_test.cpp`, for instance).  Therefore, we have to ensure that
    // the right functions are in the function map, for every type that appears
    // in any binding.
    paramsClean.functionMap =
        mlpack::bindings::tests::TestFunctionMap::FunctionMap();

    params = paramsClean;
  }

  //! Clean up any memory associated with the MainTestFixture.
  ~MainTestFixture()
  {
    // Clean any allocated memory associated with any parameters.
    CleanMemory();
  }

  /**
   * Reset the `params` object to its initial state.  After calling this method,
   * it is as though no parameters at all have been set with `SetInputParam()`.
   * Note that this does *not* clean memory associated with the current
   * parameters!  So, you may want to call `ClearMemory()` before calling this.
   */
  void ResetSettings()
  {
    // Reset the parameters.
    params = paramsClean;

    // Reset the timers too...
    timers.StopAllTimers();
    timers.Reset();
  }

  /**
   * Clean any memory associated with the `params` object.
   */
  void CleanMemory()
  {
    bindings::tests::CleanMemory(params);
  }

  /**
   * Set the input parameter `name` to have value `value`.
   */
  template<typename T>
  void SetInputParam(const std::string& name, T&& value)
  {
    params.Get<typename std::remove_reference<T>::type>(name) =
        std::forward<T>(value);
    params.SetPassed(name);
  }

 protected:
  //! Untouched "clean" parameters object, used for resetting.
  util::Params paramsClean;
  //! Parameters object, which the binding will be called with.
  util::Params params;
  //! Timers object, which the binding will be called with.
  util::Timers timers;
};

#endif
