/**
 * @file termination_policy_test.cpp
 * @author Ryan Curtin
 *
 * Tests for AMF termination policies.
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
#include <mlpack/core.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/termination_policies/max_iteration_termination.hpp>
#include <mlpack/methods/amf/update_rules/nmf_mult_div.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(TerminationPolicyTest);

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::amf;

/**
 * Simple test -- make sure termination happens after the right number of
 * iterations.
 */
BOOST_AUTO_TEST_CASE(MaxIterationTerminationTest)
{
  MaxIterationTermination mit(500);

  arma::mat x; // Just an argument to pass.
  for (size_t i = 0; i < 499; ++i)
    BOOST_REQUIRE_EQUAL(mit.IsConverged(x, x), false);

  // Should keep returning true once maximum iterations are reached.
  BOOST_REQUIRE_EQUAL(mit.IsConverged(x, x), true);
  BOOST_REQUIRE_EQUAL(mit.Iteration(), 500);
  BOOST_REQUIRE_EQUAL(mit.IsConverged(x, x), true);
  BOOST_REQUIRE_EQUAL(mit.IsConverged(x, x), true);
}

/**
 * Make sure that AMF properly terminates.
 */
BOOST_AUTO_TEST_CASE(AMFMaxIterationTerminationTest)
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  size_t r = 12;

  // Custom tighter tolerance.
  MaxIterationTermination mit(10); // Only 10 iterations.
  AMF<MaxIterationTermination,
      RandomInitialization,
      NMFMultiplicativeDivergenceUpdate> nmf(mit);
  nmf.Apply(v, r, w, h);

  BOOST_REQUIRE_EQUAL(nmf.TerminationPolicy().Iteration(), 10);
}

BOOST_AUTO_TEST_SUITE_END();
