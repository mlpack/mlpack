#include "mlpack/core/data/random_regression_generator.hpp"
#include "catch.hpp"
#include "mlpack.hpp"

using namespace mlpack::data;

TEST_CASE("Generate linear regression data - Valid Inputs",
  "[regressionGenerator]") {
  arma::mat X, y;
  
  // Test case 1 : Check with Normal Distribution error
  ErrorParams normalError(ErrorType::NormalDist);
  normalError.normalParams = NormalDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator1(100, 5, normalError);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 100);
  REQUIRE(X.n_cols == 5);
  REQUIRE(y.n_rows == 100);
  REQUIRE(y.n_cols == 1);

  // Test case 2 : Check with Gamma Distribution error
  ErrorParams gammaError(ErrorType::GammaDist);
  gammaError.gammaParams = GammaDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator2(100, 5, gammaError);
  REQUIRE_NOTHROW(generator2.GenerateData(X, y));
  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 100);
  REQUIRE(X.n_cols == 5);
  REQUIRE(y.n_rows == 100);
  REQUIRE(y.n_cols == 1);

}

TEST_CASE("Generate linear regression data - Invalid Inputs", 
  "[regressionGenerator]") {
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

// Write your test cases here
TEST_CASE("Generate linear regression data - Mutli-target ", 
  "[regressionGenerator]") {
  arma::mat X, y;

  // Test case 1 : Check with normal Distribution error
  ErrorParams normalError(ErrorType::NormalDist);
  normalError.normalParams = NormalDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator1(100, 5, normalError, 5);
  REQUIRE_NOTHROW(generator1.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 100);
  REQUIRE(X.n_cols == 5);
  REQUIRE(y.n_rows == 100);
  REQUIRE(y.n_cols == 5);

  // Test case 2 : Check with Gamma Distribution error
  ErrorParams gammaError(ErrorType::GammaDist);
  gammaError.gammaParams = GammaDistParams(3.0, 4.0);
  
  RegressionDataGenerator generator2(100, 5, gammaError, 5);
  REQUIRE_NOTHROW(generator2.GenerateData(X, y));

  // Check the correctness of the generated data
  REQUIRE(X.n_rows == 100);
  REQUIRE(X.n_cols == 5);
  REQUIRE(y.n_rows == 100);
  REQUIRE(y.n_cols == 5);
}
