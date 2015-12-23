/**
 * @file mlpack_test.cpp
 *
 * Simple file defining the name of the overall test for mlpack, and set up
 * global test fixture for each test. Each individual test is contained in
 * its own file.
 */
#define BOOST_TEST_MODULE mlpackTest

#include <mlpack/core/util/log.hpp>

#include <boost/version.hpp>

// We only need to do this for old Boost versions.
#if BOOST_VERSION < 103600
  #define BOOST_AUTO_TEST_MAIN
#endif

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

/**
 * Provide a global fixture for each test.
 *
 * A global fixture is expected to be implemented as a class where the class
 * constructor serves as a setup method and class destructor serves as teardown
 * method.
 *
 * By default, Log::objects should have their output redirected, otherwise
 * the UTF test output would be drowned out by Log::Debug and Log::Warn
 * messages.
 *
 * For more detailed test output, set the CMake flag TEST_VERBOSE=ON.
 */
struct GlobalFixture
{
  GlobalFixture()
  {
    #ifndef TEST_VERBOSE
      #ifdef DEBUG
        mlpack::Log::Debug.ignoreInput = true;
      #endif

      mlpack::Log::Info.ignoreInput = true;
      mlpack::Log::Warn.ignoreInput = true;
    #endif
  }
};

BOOST_GLOBAL_FIXTURE(GlobalFixture);
