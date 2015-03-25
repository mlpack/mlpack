/**
 * @file log_test.cpp
 * @author Marcus Edel
 *
 * Test of the mlpack log class.
 **/

#include <mlpack/core.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;

BOOST_AUTO_TEST_SUITE(LogTest);

/**
 * Simple log assert test. Be careful the test halts the program execution, so
 * run the test at the end of all other tests.
 */
BOOST_AUTO_TEST_CASE(LogAssertConditionTest)
{
  // Only do anything for Assert() if in debugging mode.
  #ifdef DEBUG
      // If everything goes well we reach the boost test condition which is
      // always true by simplicity's sake.
      Log::Assert(true, "test");
      BOOST_REQUIRE_EQUAL(1, 1);

      // The test case should halt the program execution and prints a custom
      // error message. Since the program is halted we should never reach the
      // boost test condition which is always false by simplicity's sake.
      // Log::Assert(false, "test");
      // BOOST_REQUIRE_EQUAL(1, 0);
  #endif
}

BOOST_AUTO_TEST_SUITE_END();
