/**
 * @file random_regression_generator_test.hpp
 * @author Ali Hossam
 *
 * Unit tests for Random Regression Generator.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack.hpp>
#include "catch.hpp"

using namespace mlpack;
using namespace mlpack::data;
using namespace arma;

TEST_CASE("Generate linear regression data - Valid Matrix type", 
          "[regressionGenerator]") 
{
  // Test case 1 : Test with Double precision matrix
  arma::mat X, y;
  
  RegressionDataGenerator<> generator1(100, 5);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  // Test case 2 : Test with Single precision matrix
  arma::fmat Xf, yf;  
  REQUIRE_NOTHROW(generator1.GenerateData(Xf, yf));
}

TEST_CASE("Generate linear regression data - Valid Noise Distribution Input", 
          "[regressionGenerator]") 
{ 
  arma::mat X, y;
  
  // Test case 1 : Check w  ith Gaussian Distribution (default)
  RegressionDataGenerator<> generator1(100, 5);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 5);
  REQUIRE(X.n_cols == 100);
  REQUIRE(y.n_rows == 1);
  REQUIRE(y.n_cols == 100);

  // Test case 2 : Check with Gamma Distribution
  RegressionDataGenerator<GammaDistribution>generator2(
      100, 5, GammaDistribution(vec("2"), vec("3")));
  REQUIRE_NOTHROW(generator2.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 5);
  REQUIRE(X.n_cols == 100);
  REQUIRE(y.n_rows == 1);
  REQUIRE(y.n_cols == 100);

  // Test case 3 : Check with Laplace Distribution
  RegressionDataGenerator<LaplaceDistribution>generator3(
      100, 5, LaplaceDistribution(vec("2"), 3));
  REQUIRE_NOTHROW(generator3.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 5);
  REQUIRE(X.n_cols == 100);
  REQUIRE(y.n_rows == 1);
  REQUIRE(y.n_cols == 100);

}

TEST_CASE("Generate linear regression data - Invalid Inputs", 
          "[regressionGenerator]") 
{
  // Test case 1: Invalid sparsity
  REQUIRE_THROWS_AS(
      RegressionDataGenerator<>(
          100, 5, GaussianDistribution(vec("0"), vec("1")), 1, 0.5, 1.5),
      std::invalid_argument);

  // Test case 2: Invalid outliers fraction
  REQUIRE_THROWS_AS(
      RegressionDataGenerator<>(
          100, 5, GaussianDistribution(vec("0"), vec("1")), 1, 0.5, 0.5, 1.5),
      std::invalid_argument);
}

TEST_CASE("Generate linear regression data - Mutli-target ", 
          "[regressionGenerator]") 
{
  arma::mat X, y;

  RegressionDataGenerator<> generator1(
      100, 5, GaussianDistribution(vec("0"), vec("1")), 5);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 5);
  REQUIRE(X.n_cols == 100);
  REQUIRE(y.n_rows == 5);
  REQUIRE(y.n_cols == 100);
}

TEST_CASE("Generate linear regression data - Perfect linear model",
          "[regressionGenerator]") 
{
  arma::mat X, y, y_pred;
  
  RegressionDataGenerator<> generator1(100, 5);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  mlpack::regression::LinearRegression<> lr;
  lr.Train(X, y);
  lr.Predict(X, y_pred);

  // Compute residuals mean
  arma::mat residuals = y - y_pred;
  double mean = arma::mean(arma::mean(residuals, 1));

  // Check if the mean is close to 0 (within a margin)
  INFO("Computed Mean: " << mean);
  REQUIRE(mean == Approx(0.0).margin(1e-5));

}