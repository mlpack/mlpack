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
//#include <mlpack/core/optimizers/sgd/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(PSGDTest);

BOOST_AUTO_TEST_CASE(SimpleSGDTestFunction)
{
  PSGDTestFunction f;
  ParallelSGD<PSGDTestFunction> s(f, 0.0003,   2000000, 1e-9);

  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(coordinates);
  //std::cout<<"result ="<<result<<" "<<coordinates[0]<<" "<<coordinates[1]<<" "<<coordinates[2]<<std::endl;
  BOOST_REQUIRE_CLOSE(result, -1.0, 0.05);
  BOOST_REQUIRE_SMALL(coordinates[0], 1e-3);
  BOOST_REQUIRE_SMALL(coordinates[1], 1e-7);
  BOOST_REQUIRE_SMALL(coordinates[2], 1e-7);
}

/*
BOOST_AUTO_TEST_CASE(BoothsFunction)
{
  BoothsFunction f;
  ParallelSGD<BoothsFunction> s(f, 0.0003, 500000, 1e-9);
  arma::mat coordinates = f.GetInitialPoint();
  double result = s.Optimize(coordinates);

  std::cout<<"result ="<<result<<" "<<coordinates[0]<<" "<<coordinates[1]<<std::endl;
  BOOST_REQUIRE_SMALL(result, 1e-3);
  BOOST_REQUIRE_CLOSE(coordinates[0], (double) 1.0, 1e-3);
  BOOST_REQUIRE_CLOSE(coordinates[1], (double) 3.0, 1e-3);

}
*/



BOOST_AUTO_TEST_CASE(GeneralizedRosenbrockTest)
{
  // Loop over several variants.
  for (size_t i = 10; i < 11; i += 5)
  {
    // Create the generalized Rosenbrock function.
    GeneralizedRosenbrockFunction f(i);

    ParallelSGD<GeneralizedRosenbrockFunction> s(f, 0.001,50000, 1e-15);

    arma::mat coordinates = f.GetInitialPoint();
    double result = s.Optimize(coordinates);
//    cout<<" result"<<result<<endl;
    BOOST_REQUIRE_SMALL(result, 1e-10);
    for (size_t j = 0; j < i; ++j)
    {
      //std::cout<<coordinates[j]<<std::endl;
      BOOST_REQUIRE_CLOSE(coordinates[j], (double) 1.0, 1e-3);
    }
  }
}



BOOST_AUTO_TEST_SUITE_END();



