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
 *
 * This file is part of mlpack 2.0.2.
 *
 * mlpack is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * mlpack is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * mlpack.  If not, see <http://www.gnu.org/licenses/>.
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
