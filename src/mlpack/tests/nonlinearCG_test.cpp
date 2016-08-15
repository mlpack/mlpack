#include <mlpack/core.hpp>
#include <mlpack/core/optimizers/cg/cg.hpp>
#include <mlpack/core/optimizers/cg/test_function.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace std;
using namespace arma;
using namespace mlpack;
using namespace mlpack::optimization;
using namespace mlpack::optimization::test;

BOOST_AUTO_TEST_SUITE(CGTest);

BOOST_AUTO_TEST_CASE(SimpleCGTestFunction1)
{
  //mlpack::Log::Info.ignoreInput = false;
  CGTestFunction1 f;
  mat coordinates = f.GetInitialPoint();

  arma::mat con_gradient;
  f.gradient(coordinates,con_gradient);
  NonlinearCG<CGTestFunction1> s(f,10000,1e-9);
  
  double result = s.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(coordinates[0],0.5,0.05);
  BOOST_REQUIRE_CLOSE(coordinates[1],0.5,0.05);
  BOOST_REQUIRE_CLOSE(result,12,0.05);

}



BOOST_AUTO_TEST_CASE(SimpleCGTestFunction2)
{
  //mlpack::Log::Info.ignoreInput = false;
  CGTestFunction2 f;
  mat coordinates = f.GetInitialPoint();

  arma::mat con_gradient;
  f.gradient(coordinates,con_gradient);
  NonlinearCG<CGTestFunction2> s(f,10000,1e-9);
  

  double result = s.Optimize(coordinates);

  BOOST_REQUIRE_CLOSE(coordinates[0],2,0.05);
  BOOST_REQUIRE_CLOSE(coordinates[1],3,0.05);
  BOOST_REQUIRE_SMALL(result, 1e-3);

}

BOOST_AUTO_TEST_SUITE_END();

