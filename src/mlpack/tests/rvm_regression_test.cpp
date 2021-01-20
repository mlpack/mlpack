#include <iostream>
// Includes all relevant components of mlpack.

#include <math.h>
#include <ctime>

#include <mlpack/core/data/load.hpp>

#include <boost/test/unit_test.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>

#include <mlpack/methods/rvm_regression/rvm_regression.hpp>

using namespace mlpack;
using namespace mlpack::regression;

BOOST_AUTO_TEST_SUITE(RVMRegressionTest);

void GenerateProblem(arma::mat& X,
                     arma::rowvec& y,
                     size_t nPoints,
                     size_t nDims,
                     float sigma = 0.0)
{
  X = arma::randn(nDims, nPoints);
  arma::colvec omega = arma::randn(nDims);
  omega(0) = 0;
  omega(1) = 0;
  // Compute y and add noise.
  y = omega.t() * X + arma::randn(nPoints).t() * sigma;
}

// Ensure that predictions are close enough to the target
// for a free noise dataset in ard mode and with linear kernel.
BOOST_AUTO_TEST_CASE(RVMRegressionRegressionTest)
{
  arma::mat matX;
  arma::rowvec y, predictions;

  GenerateProblem(matX, y, 200, 10);

  // RVM with linear kernel.
  kernel::LinearKernel kernel;
  RVMRegression<kernel::LinearKernel> rvmLinear(kernel, false, false, false);
  rvmLinear.Train(matX, y);
  rvmLinear.Predict(matX, predictions);

  // Check the predictions are close enough to the targets.
  for (size_t i = 0; i < y.size(); ++i)
    BOOST_REQUIRE_CLOSE(predictions[i], y[i], 1e-6);

  // ARD Regression.
  RVMRegression<kernel::LinearKernel> rvmArd(kernel, false, false, false);
  rvmArd.Train(matX, y);
  rvmArd.Predict(matX, predictions);

  // Check the predictions are close enough to the targets.
  for (size_t i = 0; i < y.size(); ++i)
    BOOST_REQUIRE_CLOSE(predictions[i], y[i], 1e-6);

  // Check that the estimated variance is zero.
  BOOST_REQUIRE_SMALL(rvmArd.Variance(), 1e-6);
  BOOST_REQUIRE_SMALL(rvmLinear.Variance(), 1e-6);
}

// Verify centerData and scaleData equal false do not affect the solution.
BOOST_AUTO_TEST_CASE(TestCenter0ScaleData0)
{
  arma::mat matX;
  arma::rowvec y;
  size_t nDims = 30, nPoints = 100;

  GenerateProblem(matX, y, nPoints, nDims, 0.5);

  kernel::LinearKernel kernel;
  RVMRegression<kernel::LinearKernel> estimator(kernel, false, false, false);

  estimator.Train(matX, y);

  // Check dataOffset is empty.
  BOOST_REQUIRE(estimator.DataOffset().n_elem == 0);

  // To be neutral responseOffset must be 0.
  BOOST_REQUIRE(estimator.ResponsesOffset() == 0);

  // Check dataScale is empty.
  BOOST_REQUIRE(estimator.DataScale().n_elem == 0);
}

// Verify that centering and normalization are correct.
BOOST_AUTO_TEST_CASE(TestCenterDataTrueScaleDataTrue)
{
  arma::mat matX;
  arma::rowvec y;
  size_t nDims = 5, nPoints = 100;
  GenerateProblem(matX, y, nPoints, nDims, 0.5);

  kernel::LinearKernel kernel;
  RVMRegression<kernel::LinearKernel> estimator(kernel, true, true, true);

  estimator.Train(matX, y);

  arma::colvec xMean = arma::mean(matX, 1);
  arma::colvec xStd = arma::stddev(matX, 1, 1);
  // Keep the actiive basis functions only.
  xMean = xMean(estimator.ActiveSet());
  xStd = xStd(estimator.ActiveSet());
  double yMean = arma::mean(y);

  BOOST_REQUIRE_SMALL((double) abs(sum(estimator.DataOffset() - xMean)), 1e-6);
  BOOST_REQUIRE_SMALL((double) abs(sum(estimator.DataScale() - xStd)), 1e-6);
  BOOST_REQUIRE_CLOSE(estimator.ResponsesOffset(), yMean, 1e-6);
}

// Make sure a model with center ans scale option set is different than a model
// without it set.
BOOST_AUTO_TEST_CASE(OptionsMakeModelDifferent)
{
  arma::mat matX;
  arma::rowvec y;
  size_t nDims = 10, nPoints = 100;
  GenerateProblem(matX, y, nPoints, nDims, 0.5);

  kernel::LinearKernel kernel;
  RVMRegression<kernel::LinearKernel> rvm(kernel, false, false, true),
      rvmC(kernel, true, false, true), rvmCS(kernel, true, true, true);

  rvm.Train(matX, y);
  rvmC.Train(matX, y);
  rvmCS.Train(matX, y);

  for (size_t i = 0; i < nDims; ++i)
    BOOST_REQUIRE((rvm.Omega()(i) != rvmC.Omega()(i)) &&
                  (rvm.Omega()(i) != rvmCS.Omega()(i)) &&
                  (rvmC.Omega()(i) != rvmCS.Omega()(i)));
}

// Check that Train() does not fail with two colinear vectors.
BOOST_AUTO_TEST_CASE(SingularMatix)
{
  arma::mat matX;
  arma::rowvec y;

  GenerateProblem(matX, y, 200, 10);
  // Now the first and the second rows are indentical.
  matX.row(1) = matX.row(0);

  kernel::LinearKernel kernel;
  RVMRegression<kernel::LinearKernel> estimator(kernel, true, false, true);
  estimator.Train(matX, y);
}

// Check that std are well computed/coherent. At least higher than the
// estimated predictive variance.
BOOST_AUTO_TEST_CASE(PredictiveUncertainties)
{
  arma::mat matX;
  arma::rowvec y;

  GenerateProblem(matX, y, 100, 10, 1);

  // ARD case.
  kernel::LinearKernel kernel;
  RVMRegression<kernel::LinearKernel> estimator(kernel, true, false, true);
  estimator.Train(matX, y);

  arma::rowvec responses, std;
  estimator.Predict(matX, responses, std);
  const double estStd = sqrt(estimator.Variance());

  for (size_t i = 0; i < matX.n_cols; i++)
    BOOST_REQUIRE_GT(std[i], estStd);

  // Check that the estimated variance is close to 1.
  BOOST_REQUIRE_CLOSE(estStd, 1, 20);

  // Liner Kernel case.
  estimator = RVMRegression<kernel::LinearKernel>(kernel, true, false, false);

  for (size_t i = 0; i < matX.n_cols; i++)
    BOOST_REQUIRE_GT(std[i], estStd);

  // Check that the estimated variance is close to 1.
  BOOST_REQUIRE_CLOSE(estStd, 1, 20);

}

BOOST_AUTO_TEST_SUITE_END();
