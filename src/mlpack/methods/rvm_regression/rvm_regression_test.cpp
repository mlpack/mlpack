#include <iostream>
// Includes all relevant components of mlpack.

#include <math.h>
#include <ctime>

#include <mlpack/core/data/load.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/methods/lars/lars.hpp>
#include <mlpack/methods/linear_regression/linear_regression.hpp>

#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE Mytest
#include <boost/test/unit_test.hpp>


#include "rvmr.hpp"

using namespace mlpack;
using namespace rvmr;


BOOST_AUTO_TEST_CASE(RVMegressionTest)
{
  // First, load the data.
  arma::mat Xtrain, Xtest;
  arma::rowvec ytrain, ytest;
  double RMSETRAIN = 0.112415, RMSETEST = 0.171325;


  std::cout<< "Synthetic dataset.\n" 
  	   << "Only the first ten features are non equal to 0."
  	   << std::endl;
  data::Load("./data/synth_train.csv", Xtrain, false, true);  
  data::Load("./data/synth_test.csv", Xtest, false, true);  
  data::Load("./data/synth_y_train.csv", ytrain, false, true);
  data::Load("./data/synth_y_test.csv", ytest, false, true);  

  // Instanciate and train the estimator
  RVMR<kernel::LinearKernel> estimator(true, false);
  estimator.Train(Xtrain, ytrain);

  // Check if the RMSE are still equal to the previously fixed values
  BOOST_REQUIRE_SMALL(estimator.Rmse(Xtrain,ytrain) - RMSETRAIN, 0.05);
  BOOST_REQUIRE_SMALL(estimator.Rmse(Xtest,ytest) - RMSETEST, 0.05);

  //FIX ME TRain a LARS estimator
  arma::vec predTestLars, solution;
  regression::LARS lars(true);
  lars.Train(Xtrain, ytrain.t(), solution);
  lars.Predict(Xtest, predTestLars);
  predTestLars.print();
  arma::rowvec predTestRvm;
  estimator.Predict(Xtest, predTestRvm);
  std::cout << "\n" << std::endl;
  predTestRvm.print();
  std::cout << "end of the code" << std::endl;
}
