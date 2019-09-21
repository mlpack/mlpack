#include <iostream>
// Includes all relevant components of mlpack.

#include <math.h>
#include <ctime>

#include <mlpack/core/data/load.hpp>

// #define BOOST_TEST_DYN_LINK 
// #define BOOST_TEST_MODULE BayesianRidgeTest
#include <boost/test/unit_test.hpp>

#include <mlpack/methods/bayesian_ridge/bayesian_ridge.hpp>

using namespace mlpack::regression;
using namespace mlpack::data;
using namespace std;
using namespace arma;

BOOST_AUTO_TEST_SUITE(BayesianRidgeTest);

void GenerateProblem(arma::mat& X,
		     arma::rowvec& y,
		     size_t nPoints,
		     size_t nDims,
		     float sigma=0.0)
{
  arma_rng::set_seed(4);
    
  X = arma::randn(nDims, nPoints);
  arma::colvec omega = arma::randn(nDims);
  arma::colvec noise = arma::randn(nPoints) * sigma;
  y = (omega.t() * X);
  y += noise;
}

BOOST_AUTO_TEST_CASE(BayesianRidgeRegressionTest)
{
  // First, load the data.
  mat Xtrain, Xtest;
  rowvec ytrain, ytest;

  // The RMSE are set according to the results obtained by the current
  // implementation on the following dataset.
  // y = Xw + noise, noise->Normal(0,1/beta), where beta=40 is the
  // precision and w vector/solution to recover with the seven first elements
  // non zero.
  const double RMSETRAIN = 0.14507, RMSETEST = 0.17961;
  
  Load("reg_x_train.csv", Xtrain, false, true);  
  Load("reg_x_test.csv", Xtest, false, true);  
  Load("reg_y_train.csv", ytrain, false, true);
  Load("reg_y_test.csv", ytest, false, true);  

  // Instanciate and train the estimator
  BayesianRidge estimator(true, false);
  estimator.Train(Xtrain, ytrain);

  // Check if the RMSE are still equal to the previously fixed values
  BOOST_REQUIRE_SMALL(estimator.Rmse(Xtrain,ytrain) - RMSETRAIN, 0.05);
  BOOST_REQUIRE_SMALL(estimator.Rmse(Xtest,ytest) - RMSETEST, 0.05);

}

BOOST_AUTO_TEST_CASE(TestCenterNormalize)
{
  arma::mat X;
  arma::rowvec y;
  size_t nDims = 30, nPoints = 100;
  GenerateProblem(X, y, nPoints, nDims, 0.5);

  BayesianRidge estimator(false, false);
  estimator.Train(X,y);

  // To be neutral data_offset must be all 0.
  BOOST_TEST(sum(estimator.getdata_offset()) == 0);

  // To be neutral data_scale must be all 1.
  BOOST_TEST(sum(estimator.getdata_scale()) == nDims);
}

BOOST_AUTO_TEST_CASE(ColinearTest)
{
  arma::mat X;
  arma::rowvec y;

  Load("lars_dependent_x.csv", X, false, true);
  Load("lars_dependent_y.csv", y, false, true);

  BayesianRidge estimator(false, false);
  estimator.Train(X,y);
}

BOOST_AUTO_TEST_CASE(OnePointTest)
{
  arma::mat X;
  arma::rowvec y;
  arma::rowvec predictions, std;
  double y_i, std_i;

  Load("reg_x_train.csv", X, false, true);  
  Load("reg_y_train.csv", y, false, true);
  
  BayesianRidge estimator(false, false);
  estimator.Train(X,y);

  // Predict on all the points.
  estimator.Predict(X, predictions);

  // Ensure that the single prediction from column vector are possible and
  // equal to the matrix version.
  for (size_t i = 0; i < y.size(); i++)
    {
      estimator.Predict(X.col(i), y_i);
      BOOST_REQUIRE_CLOSE(predictions(i), y_i, 1e-5); 
    }

  // Ensure that the single prediction from column vector are possible and
  // equal to the matrix version. Idem for the std.
  estimator.Predict(X, predictions, std);
  for (size_t i = 0; i < y.size(); i++)
    {
      estimator.Predict(X.col(i), y_i, std_i);
      BOOST_REQUIRE_CLOSE(predictions(i), y_i, 1e-5);
      BOOST_REQUIRE_CLOSE(std(i), std_i, 1e-5); 
    }
 }

BOOST_AUTO_TEST_SUITE_END();


