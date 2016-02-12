/**
 * @file minibatch_sgd_test.cpp
 * @author Ryan Curtin
 *
 * Test file for minibatch SGD.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/minibatch_sgd/minibatch_sgd.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(MiniBatchSGDTest);

/**
 * If the batch size is 1, and we aren't shuffling, we should get the exact same
 * results as regular SGD.
 */
BOOST_AUTO_TEST_CASE(SGDSimilarityTest)
{
  SGDTestFunction f;
  SGD<SGDTestFunction> s(f, 0.0003, 5000000, 1e-9, false);
  MiniBatchSGD<SGDTestFunction> ms(f, 1, 0.0003, 5000000, 1e-9, false);

  arma::mat sCoord = f.GetInitialPoint();
  arma::mat msCoord = f.GetInitialPoint();

  const double sResult = s.Optimize(sCoord);
  const double msResult = s.Optimize(msCoord);

  BOOST_REQUIRE_CLOSE(sResult, msResult, 1e-8);
  BOOST_REQUIRE_CLOSE(sCoord[0], msCoord[0], 1e-8);
  BOOST_REQUIRE_CLOSE(sCoord[1], msCoord[1], 1e-8);
  BOOST_REQUIRE_CLOSE(sCoord[2], msCoord[2], 1e-8);
}

/*
BOOST_AUTO_TEST_CASE(SimpleSGDTestFunction)
{
  SGDTestFunction f;
  // Batch size of 3.
  MiniBatchSGD<SGDTestFunction> s(f, 3, 0.0005, 2000000, 1e-9, true);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
}
*/

BOOST_AUTO_TEST_SUITE_END();
