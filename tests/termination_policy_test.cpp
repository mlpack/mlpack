/**
 * @file tests/termination_policy_test.cpp
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
#include <mlpack/methods/amf.hpp>

#include "catch.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;

/**
 * Simple test -- make sure termination happens after the right number of
 * iterations.
 */
TEST_CASE("MaxIterationTerminationTest", "[TerminationPolicyTest]")
{
  MaxIterationTermination mit(500);

  arma::mat x; // Just an argument to pass.
  for (size_t i = 0; i < 499; ++i)
    REQUIRE(mit.IsConverged(x, x) == false);

  // Should keep returning true once maximum iterations are reached.
  REQUIRE(mit.IsConverged(x, x) == true);
  REQUIRE(mit.Iteration() == 500);
  REQUIRE(mit.IsConverged(x, x) == true);
  REQUIRE(mit.IsConverged(x, x) == true);
}

/**
 * Make sure that AMF properly terminates.
 */
TEST_CASE("AMFMaxIterationTerminationTest", "[TerminationPolicyTest]")
{
  mat w = randu<mat>(20, 12);
  mat h = randu<mat>(12, 20);
  mat v = w * h;
  size_t r = 12;

  // Custom tighter tolerance.
  MaxIterationTermination mit(10); // Only 10 iterations.
  AMF<MaxIterationTermination,
      RandomAMFInitialization,
      NMFMultiplicativeDivergenceUpdate> nmf(mit);
  nmf.Apply(v, r, w, h);

  REQUIRE(nmf.TerminationPolicy().Iteration() == 10);
}
