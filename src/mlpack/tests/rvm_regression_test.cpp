/**
 * @file tests/rvm_regression_test.cpp
 * @author Clement Mercier
 *
 * Tests for RVMRegression class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
//#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/core/kernels/linear_kernel.hpp>
#include <mlpack/core/kernels/gaussian_kernel.hpp>
#include <mlpack/methods/rvm_regression/rvm_regression.hpp>

#include "serialization.hpp"

using namespace mlpack::regression;

void GenerateProblemSparse(arma::mat& matX,
                           arma::rowvec& y,
	                   size_t nPoints,
			   size_t nDims,
			   float sigma = 0.0)
{
  matX = arma::randn(nDims, nPoints);
  arma::colvec omega = arma::zeros<arma::colvec>(nDims);
  omega.head(nDims / 5) = arma::randn(nDims / 5) * 10; 
  
  // Compute y and add noise.
  y = omega.t() * matX + arma::randn(nPoints).t() * sigma;
}

TEST_CASE("OptionsMakeRVMDifferent", "[RVMRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y, predictions;
  size_t nPoints = 200, nDims = 25;
  GenerateProblemSparse(matX, y, nPoints, nDims, 1);

  mlpack::kernel::LinearKernel kernel;
  RVMRegression<mlpack::kernel::LinearKernel> ard10(kernel);
  RVMRegression<mlpack::kernel::LinearKernel> ard11(kernel, true, true, true);
  RVMRegression<mlpack::kernel::LinearKernel> rvm10(kernel, true, false, false);
  RVMRegression<mlpack::kernel::LinearKernel> rvm11(kernel, true, true, false);
  ard10.Train(matX, y);
  ard11.Train(matX, y);
  rvm10.Train(matX, y);
  rvm11.Train(matX, y);

  for (size_t i = 0; i < nDims; ++i)
  {
    REQUIRE(ard10.Alpha()(i) != ard11.Alpha()(i));
  }  
  for (size_t i = 0; i < nPoints; ++i)
  {
    REQUIRE(rvm10.Alpha()(i) != rvm11.Alpha()(i));
  }  
}

// Check that Train() does not fail with two colinear vectors. Try it for ARD
// regression, Linear RVM and Gaussian RVM.
TEST_CASE("SingularMatrixRVM", "[RVMRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y;

  GenerateProblemSparse(matX, y, 200, 25);

  // Now the first and the second rows are indentical.
  matX.row(1) = matX.row(0);
  
  mlpack::kernel::LinearKernel linear;
  mlpack::kernel::GaussianKernel gaussian(10);

  // ARD regression.
  RVMRegression<mlpack::kernel::LinearKernel> ard11(linear, true, true, true);
  // Linear kernel RVM.
  RVMRegression<mlpack::kernel::LinearKernel>
    rvm10(linear, true, false, false);
  // Gaussian kernel RVM.
  RVMRegression<mlpack::kernel::GaussianKernel>
      rvmGaussian10(gaussian, true, false, false);

  ard11.Train(matX, y);
  rvm10.Train(matX, y);
  rvmGaussian10.Train(matX, y);
}

// Check that std are well computed/coherent. At least higher than the
// estimated predictive variance.
TEST_CASE("PredictiveUncertaintiesRVM", "[RVMRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y, responses, std;
  double estStd;
  
  GenerateProblemSparse(matX, y, 200, 40, 2.0);

  mlpack::kernel::LinearKernel linear;

  // ARD regression.
  RVMRegression<mlpack::kernel::LinearKernel> ard00(linear, false, false, true);
  // Linear kernel RVM.
  RVMRegression<mlpack::kernel::LinearKernel>
    rvm00(linear, false, false, false);

  ard00.Train(matX, y);
  rvm00.Train(matX, y);

  // ARD predict.
  ard00.Predict(matX, responses, std);
  estStd = sqrt(ard00.Variance());

  // Check that the estimated variance is close to 1.
  REQUIRE(estStd == Approx(2).epsilon(0.3));

  for (size_t i = 0; i < matX.n_cols; i++)
    REQUIRE(std[i] > estStd);

  // RVM Linear predict.
  rvm00.Predict(matX, responses, std);
  estStd = sqrt(rvm00.Variance());

  // Check that the estimated variance is close to 1.
  REQUIRE(estStd == Approx(2).epsilon(0.3));

  for (size_t i = 0; i < matX.n_cols; i++)
    REQUIRE(std[i] > estStd);
}

TEST_CASE("SerializeRVM", "[RVMRegressionTest]")
{
  arma::mat matX;
  arma::rowvec y;

  GenerateProblemSparse(matX, y, 200, 25, 2.0);

  mlpack::kernel::LinearKernel linear;

  // ARD regression.
  RVMRegression<mlpack::kernel::LinearKernel> ard(linear, true, false, true);

  ard.Train(matX, y);

  arma::rowvec beforePredictions, beforeStd;
  ard.Predict(matX, beforePredictions, beforeStd);

  // Serialization.
  RVMRegression<mlpack::kernel::LinearKernel> xmlArd, jsonArd, binArd;
  mlpack::SerializeObjectAll(ard, xmlArd, jsonArd, binArd);

  // Now check that we get the same results serializing other things.
  arma::rowvec xmlPredictions, jsonPredictions, binPredictions;
  arma::rowvec xmlStd, jsonStd, binStd;

  xmlArd.Predict(matX, xmlPredictions, xmlStd);

  jsonArd.Predict(matX, jsonPredictions, jsonStd);
  binArd.Predict(matX, binPredictions, binStd);

  for (size_t i = 0; i < y.n_elem; ++i)
  {
    REQUIRE(beforePredictions[i] * 3 == (xmlPredictions[i] +
					 jsonPredictions[i] +
					 binPredictions[i]));
    REQUIRE(beforeStd[i] * 3 == (xmlStd[i] +
				 jsonStd[i] +
				 binStd[i]));
  }
}
