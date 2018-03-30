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
#include <mlpack/core/optimizers/parallel_sgd/decay_policies/constant_step.hpp>
#include <mlpack/core/optimizers/parallel_sgd/decay_policies/exponential_backoff.hpp>
#include <mlpack/core/optimizers/parallel_sgd/sparse_test_function.hpp>
#include <mlpack/core/optimizers/problems/generalized_rosenbrock_function.hpp>

// We need some thorough testing.
#define private public
#include <mlpack/core/optimizers/parallel_sgd/parallel_sgd.hpp>
#undef private

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(ParallelSGDTest);


// These tests are only compiled if the user has specified OpenMP to be
// used.
#ifdef HAS_OPENMP

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

  for (size_t i = threadsAvailable; i > 0; --i)
  {
    omp_set_num_threads(i);

    size_t batchSize = std::ceil((float) f.NumFunctions() / i);

    ParallelSGD<ConstantStep> s(10000, batchSize, 1e-5, true, decayPolicy);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(f, coordinates);

    // The final value of the objective function should be close to the optimal
    // value, that is the sum of values at the vertices of the parabolas.
    BOOST_REQUIRE_CLOSE(result, 123.75, 0.01);

    // The co-ordinates should be the vertices of the parabolas.
    BOOST_REQUIRE_CLOSE(coordinates[0], 2, 0.02);
    BOOST_REQUIRE_CLOSE(coordinates[1], 1, 0.02);
    BOOST_REQUIRE_CLOSE(coordinates[2], 1.5, 0.02);
    BOOST_REQUIRE_CLOSE(coordinates[3], 4, 0.02);
  }
}

/**
 * When run with a single thread, parallel SGD should be identical to normal
 * SGD.
 */
BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    ConstantStep decayPolicy(0.001);

    ParallelSGD<ConstantStep> s(0, f.NumFunctions(), 1e-12, true, decayPolicy);

    arma::mat coordinates = f.GetInitialPoint();

    omp_set_num_threads(1);
    double result = s.Optimize(f, coordinates);

    BOOST_REQUIRE_SMALL(result, 1e-8);
    for (size_t j = 0; j < i; ++j)
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 0.01);
  }
}

#endif

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
