/**
 * @file cmaes_test.cpp
 * @author Ryan Curtin
 *
 * Test file for CMAES (Covariance Matrix Adaptation Evolution Strategy).
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/cmaes/cmaes.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/cmaes/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(CMAESTest);

BOOST_AUTO_TEST_CASE(SimpleCMAESTestFunction)
{
  cmaesTestFunction func;
  
  // rest parameter calculated by default 
  // or else we can call setters to change the parameters
 
  CMAES<cmaesTestFunction, double> fo(func); 

  double arr[3];
  double result = fo.Optimize(func, arr);

  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(arr[0], 1e-3);
  BOOST_REQUIRE_SMALL(arr[1], 1e-7);
  BOOST_REQUIRE_SMALL(arr[2], 1e-7);

}

BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  // Loop over several variants.
  for (size_t i = 10; i < 50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    CMAES<GeneralizedRosenbrockFunction, double> fo(f);

    double arr[i];
    double result = fo.Optimize(f, arr);

    BOOST_REQUIRE_SMALL(result, 1e-10);
    for (size_t j = 0; j < i; ++j)
      BOOST_REQUIRE_CLOSE(arr[j], (double) 1.0, 1e-3);
  }
}

BOOST_AUTO_TEST_SUITE_END();
