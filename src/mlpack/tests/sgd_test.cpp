/**
 * @file sgd_test.cpp
 * @author Ryan Curtin
 *
 * Test file for SGD (stochastic gradient descent).
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(SGDTest);

BOOST_AUTO_TEST_CASE(SimpleSGDTestFunction)
{
  SGDTestFunction f;
  SGD<SGDTestFunction> s(f, 0.0003, 5000000, 1e-9);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(coordinates);
  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
}




BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    SGD<GeneralizedRosenbrockFunction> s(f, 0.001,0, 1e-15);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(coordinates);
    BOOST_REQUIRE_SMALL(result, 1e-10);
    for (size_t j = 0; j < i; ++j)
    {
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-3);
    }
  }
}



BOOST_AUTO_TEST_SUITE_END();



