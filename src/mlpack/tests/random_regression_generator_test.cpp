#include "mlpack/core/data/random_regression_generator.hpp"
#include "catch.hpp"
#include "mlpack.hpp"

using namespace mlpack::data;
TEST_CASE("Generate linear regression data - Valid Matrix type", 
          "[regressionGenerator]") 
{
  // Test case 1 : Test with Double precision matrix
  arma::mat X, y;
  
  ErrorParams normalError(ErrorType::NormalDist);
  normalError.normalParams = NormalDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator1(100, 5, normalError);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  // Test case 2 : Test with Single precision matrix
  arma::fmat Xf, yf;  
  REQUIRE_NOTHROW(generator1.GenerateData(Xf, yf));
}

TEST_CASE("Generate linear regression data - Valid Error Input", 
          "[regressionGenerator]") 
{ 
  arma::mat X, y;
  
  // Test case 1 : Check with Normal Distribution error
  ErrorParams normalError(ErrorType::NormalDist);
  normalError.normalParams = NormalDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator1(100, 5, normalError);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 5);
  REQUIRE(X.n_cols == 100);
  REQUIRE(y.n_rows == 1);
  REQUIRE(y.n_cols == 100);

  // Test case 2 : Check with Gamma Distribution error
  ErrorParams gammaError(ErrorType::GammaDist);
  gammaError.gammaParams = GammaDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator2(100, 5, gammaError);
  REQUIRE_NOTHROW(generator2.GenerateData(X, y));
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
  arma::mat X, y;
  ErrorParams normalError(ErrorType::NormalDist);
  normalError.normalParams = NormalDistParams(3.0, 4.0);
  
  REQUIRE_THROWS_AS(RegressionDataGenerator(100, 5, normalError, 1, 0.5, 1.5),
                    std::invalid_argument);

  // Test case 2: Invalid outliers fraction
  REQUIRE_THROWS_AS(RegressionDataGenerator(100, 5, normalError, 1, 0.5, 0.5, 
                    1.5), std::invalid_argument);
}

TEST_CASE("Generate linear regression data - Mutli-target ", 
          "[regressionGenerator]") 
{
  arma::mat X, y;

  // Test case 1 : Check with normal Distribution error
  ErrorParams normalError(ErrorType::NormalDist);
  normalError.normalParams = NormalDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator1(100, 5, normalError, 5);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 5);
  REQUIRE(X.n_cols == 100);
  REQUIRE(y.n_rows == 5);
  REQUIRE(y.n_cols == 100);

  // Test case 2 : Check with Gamma Distribution error
  ErrorParams gammaError(ErrorType::GammaDist);
  gammaError.gammaParams = GammaDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator2(100, 5, gammaError, 5);
  REQUIRE_NOTHROW(generator2.GenerateData(X, y));

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
  
  ErrorParams normalError(ErrorType::NormalDist);
  normalError.normalParams = NormalDistParams(0, 1);
  
  RegressionDataGenerator generator1(100, 5, normalError);
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