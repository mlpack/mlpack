/**
 * @file tests/main_tests/tsne_test.cpp
 * @author Kiner Shah
 *
 * Test CLI binding for t-SNE.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "TSNE";

#include <mlpack/core.hpp>
#include <mlpack/methods/tsne/tsne_main.cpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(TSNETestFixture);

/**
 * Check that output dimensionality matches the requested dimensionality.
 */
TEST_CASE_METHOD(TSNETestFixture, "TSNEOutputDimensionalityTest",
    "[TSNEMainTest][BindingTests]")
{
  // Create a random dataset.
  arma::mat data = arma::randu<arma::mat>(10, 100);
  
  SetInputParam("input", std::move(data));
  SetInputParam("new_dimensionality", 2);
  SetInputParam("perplexity", 30.0);
  SetInputParam("max_iterations", 100);
  SetInputParam("random_seed", 42);
  
  REQUIRE_NOTHROW(RUN_BINDING());
  
  // Check output dimensionality.
  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(output.n_rows == 2);
  REQUIRE(output.n_cols == 100);
}

/**
 * Check that different perplexity values are respected.
 */
TEST_CASE_METHOD(TSNETestFixture, "TSNEPerplexityParameterTest",
    "[TSNEMainTest][BindingTests]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  // Run with perplexity 10
  SetInputParam("input", data);
  SetInputParam("perplexity", 10.0);
  SetInputParam("max_iterations", 100);
  SetInputParam("random_seed", 42);
  
  REQUIRE_NOTHROW(RUN_BINDING());
  arma::mat output1 = params.Get<arma::mat>("output");
  
  // Reset and run with perplexity 20
  bindings::tests::CleanMemory();
  ResetSettings();
  
  SetInputParam("input", data);
  SetInputParam("perplexity", 20.0);
  SetInputParam("max_iterations", 100);
  SetInputParam("random_seed", 42);
  
  REQUIRE_NOTHROW(RUN_BINDING());
  arma::mat output2 = params.Get<arma::mat>("output");
  
  // Results should be different
  double diff = arma::norm(output1 - output2, "fro");
  REQUIRE(diff > 1e-5);
}

/**
 * Check that negative perplexity is rejected.
 */
TEST_CASE_METHOD(TSNETestFixture, "TSNENegativePerplexityTest",
    "[TSNEMainTest][BindingTests]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  SetInputParam("input", std::move(data));
  SetInputParam("perplexity", -10.0);
  
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that negative learning rate is rejected.
 */
TEST_CASE_METHOD(TSNETestFixture, "TSNENegativeLearningRateTest",
    "[TSNEMainTest][BindingTests]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  SetInputParam("input", std::move(data));
  SetInputParam("learning_rate", -100.0);
  
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that requesting more dimensions than exist is rejected.
 */
TEST_CASE_METHOD(TSNETestFixture, "TSNEExcessiveDimensionalityTest",
    "[TSNEMainTest][BindingTests]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  SetInputParam("input", std::move(data));
  SetInputParam("new_dimensionality", 10); // More than 5
  
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that early exaggeration < 1.0 is rejected.
 */
TEST_CASE_METHOD(TSNETestFixture, "TSNEInvalidEarlyExaggerationTest",
    "[TSNEMainTest][BindingTests]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  SetInputParam("input", std::move(data));
  SetInputParam("early_exaggeration", 0.5);
  
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that the same random seed produces the same results.
 */
TEST_CASE_METHOD(TSNETestFixture, "TSNERandomSeedTest",
    "[TSNEMainTest][BindingTests]")
{
  arma::mat data = arma::randu<arma::mat>(5, 50);
  
  // Run with seed 42
  SetInputParam("input", data);
  SetInputParam("random_seed", 42);
  SetInputParam("max_iterations", 100);
  
  REQUIRE_NOTHROW(RUN_BINDING());
  arma::mat output1 = params.Get<arma::mat>("output");
  
  // Reset and run with same seed
  bindings::tests::CleanMemory();
  ResetSettings();
  
  SetInputParam("input", data);
  SetInputParam("random_seed", 42);
  SetInputParam("max_iterations", 100);
  
  REQUIRE_NOTHROW(RUN_BINDING());
  arma::mat output2 = params.Get<arma::mat>("output");
  
  // Results should be very similar
  REQUIRE(arma::approx_equal(output1, output2, "both", 1e-5, 1e-5));
}

/**
 * Check that default parameters work.
 */
TEST_CASE_METHOD(TSNETestFixture, "TSNEDefaultParametersTest",
    "[TSNEMainTest][BindingTests]")
{
  arma::mat data = arma::randu<arma::mat>(10, 100);
  
  SetInputParam("input", std::move(data));
  SetInputParam("random_seed", 42);
  
  REQUIRE_NOTHROW(RUN_BINDING());
  
  arma::mat output = params.Get<arma::mat>("output");
  REQUIRE(output.n_rows == 2); // Default dimensionality
  REQUIRE(output.n_cols == 100);
}
