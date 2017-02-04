/**
 * @file termination_policy_test.cpp
 * @author Ryan Curtin
 *
 * Tests for AMF termination policies.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/amf/amf.hpp>
#include <mlpack/methods/amf/termination_policies/max_iteration_termination.hpp>
#include <mlpack/methods/amf/update_rules/nmf_mult_div.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

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
