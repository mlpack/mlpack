/*
 * *
 * @file parallel_sgd_test.cpp
 * @author Ranjan Mondal
 *
 * Test file for  parallel stochastic gradient descent).
 */
#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/parallel_sgd/sgdp.hpp>
#include <mlpack/core/optimizers/lbfgs/test_functions.hpp>
#include <mlpack/core/optimizers/parallel_sgd/test_function.hpp>

#include <mlpack/core/optimizers/sgd/sgd.hpp>
#include <mlpack/methods/logistic_regression/logistic_regression.hpp>


#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

  
BOOST_AUTO_TEST_SUITE(PSGDTest);

BOOST_AUTO_TEST_CASE(LogisticRegressionPSGDSimpleTest)
{
  arma::mat data("1 2 3;""1 2 3;");
  arma::Row<size_t> responses("1 1 0");
  
  LogisticRegressionFunction<> lrf(data, responses, 0.001);
  ParallelSGD<LogisticRegressionFunction<>> psgd(lrf, 0.005,500000, 1e-10,false);

  LogisticRegression<> lr(psgd);
  arma::vec sigmoids = 1 / (1 + arma::exp(-lr.Parameters()[0] - data.t() * lr.Parameters().subvec(1, lr.Parameters().n_elem - 1)));
    BOOST_REQUIRE_CLOSE(sigmoids[0], 1.0, 7.0);
    BOOST_REQUIRE_CLOSE(sigmoids[1], 1.0, 14.0);
    BOOST_REQUIRE_SMALL(sigmoids[2], 0.1);
}



/*
  The  function f(x,y)=(x+2y-7)^2 +(2x+y-5)^2
  minimum value of this function is 0,  at x=1  y=3;
*/



BOOST_AUTO_TEST_CASE(BoothsFunctionTest)
{
  BoothsFunction f;
  ParallelSGD<BoothsFunction> s(f, 0.0003, 5000000, 1e-9,false);
  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(coordinates);
  BOOST_REQUIRE_SMALL(result, 1e-3);
  BOOST_REQUIRE_CLOSE(coordinates[0], (double) 1.0, 0.1);
  BOOST_REQUIRE_CLOSE(coordinates[1], (double) 3.0, 0.1);

}


BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  // Loop over several variants.
  for (size_t i = 10; i <50; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    ParallelSGD<GeneralizedRosenbrockFunction> s(f, 0.001,0, 1e-15,false);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(coordinates);
    BOOST_REQUIRE_SMALL(result, 1e-4);
    for (size_t j = 0; j < i; ++j)
    {
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 2);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();



