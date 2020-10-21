/**
 * @file tests/log_test.cpp
 * @author Marcus Edel
 *
 * Test of the mlpack log class.
 **/

#include <mlpack/core.hpp>

#include "catch.hpp"

using namespace mlpack;

/**
 * Simple log assert test. Be careful the test halts the program execution, so
 * run the test at the end of all other tests.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
TEST_CASE("LogAssertConditionTest", "[LogTest]")
{
  // Only do anything for Assert() if in debugging mode.
  #ifdef DEBUG
      // If everything goes well we reach the Catch2 test condition which is
      // always true by simplicity's sake.
      Log::Assert(true, "test");
      REQUIRE(1 == 1);

      // The test case should halt the program execution and prints a custom
      // error message. Since the program is halted we should never reach the
      // Catch2 test condition which is always false by simplicity's sake.
      // Log::Assert(false, "test");
      // REQUIRE(1 == 0);
  #endif
}
