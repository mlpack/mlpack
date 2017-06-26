/**
 * @file parallel_sgd_test.cpp
 * @author Shikhar Bhardwaj
 *
 * Test file for Parallel SGD.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/parallel_sgd/parallel_sgd.hpp>
#include <mlpack/core/optimizers/parallel_sgd/decay_policies/constant_step.hpp>
#include <mlpack/core/optimizers/parallel_sgd/decay_policies/exponential_backoff.hpp>
#include <mlpack/core/optimizers/parallel_sgd/sparse_test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(ParallelSGDTest);

/**
 * Test the correctness of the Parallel SGD implementation using a specified
 * sparse test function, with guaranteed disjoint updates between different
 * threads.
 */
BOOST_AUTO_TEST_CASE(SimpleParallelSGDTest)
{
  SparseTestFunction f;

  ConstantStep decayPolicy(0.4);

  // The batch size for this test should be chosen according to the threads
  // available on the system. If the update does not touch each datapoint, the
  // test will fail.

  size_t threadsAvailable = omp_get_max_threads();
  size_t batchSize = std::ceil((float) f.NumFunctions() / threadsAvailable);

  ParallelSGD<SparseTestFunction, ConstantStep> s(f, 10000, batchSize, 1e-5,
      decayPolicy);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(coordinates);

  // The final value of the objective funtion should be close to the optimal
  // value, that is the sum of values at the vertices of the parabolae.
  BOOST_REQUIRE_CLOSE(result, 123.75, 0.01);

  // The co-ordinates should be the vertices of the parabolae.
  BOOST_REQUIRE_CLOSE(coordinates[0], 2, 0.02);
  BOOST_REQUIRE_CLOSE(coordinates[1], 1, 0.02);
  BOOST_REQUIRE_CLOSE(coordinates[2], 1.5, 0.02);
  BOOST_REQUIRE_CLOSE(coordinates[3], 4, 0.02);
}

/**
 * Test the correctness of the Exponential backoff stepsize decay policy.
 */
BOOST_AUTO_TEST_CASE(ExponentialBackoffDecayTest)
{
  ExponentialBackoff decayPolicy(100, 100, 0.9);

  // At the first iteration, stepsize should be unchanged
  BOOST_REQUIRE_EQUAL(decayPolicy.StepSize(1), 100);
  // At the 99th iteration, stepsize should be unchanged
  BOOST_REQUIRE_EQUAL(decayPolicy.StepSize(99), 100);
  // At the 100th iteration, stepsize should be changed
  BOOST_REQUIRE_EQUAL(decayPolicy.StepSize(100), 90);
  // At the 210th iteration, stepsize should be unchanged
  BOOST_REQUIRE_EQUAL(decayPolicy.StepSize(210), 90);
  // At the 211th iteration, stepsize should be changed
  BOOST_REQUIRE_EQUAL(decayPolicy.StepSize(211), 81);
}

BOOST_AUTO_TEST_SUITE_END();
