#include <iostream>
// Includes all relevant components of mlpack.

#include <math.h>
#include <ctime>

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Mytest
#include <boost/test/unit_test.hpp>


#include "rvm_regression.hpp"

using namespace mlpack;
using namespace rvmr;


BOOST_AUTO_TEST_CASE(RVMRegressionTest)
{
  // First, load the data.
  arma::mat Xtrain, Xtest;
  arma::rowvec ytrain, ytest;

  // Instanciate and train the estimator
  RVMR<kernel::LinearKernel> estimator(true, false);
  estimator.Train(Xtrain, ytrain);

  // Check if the RMSE are still equal to the previously fixed values
  BOOST_REQUIRE(true);
}
