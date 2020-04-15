/**
 * @file bayesian_ridge_test.cpp
 * @author Clement Mercier
 *
 * Test mlpackMain() of pca_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "BayesianRidge";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/bayesian_ridge/bayesian_ridge_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct BRTestFixture
{
 public:
  BRTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~BRTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(BayesianRidgeMainTest, BRTestFixture);

/**
 * Check the center and scale options.
 */
BOOST_AUTO_TEST_CASE(BRCenter0Scale0)
{
  int n = 50, m = 4;
  arma::mat X = arma::randu<arma::mat>(n, m);
  arma::colvec omega = arma::randu<arma::colvec>(m);
  arma::mat y =  X * omega;

  SetInputParam("input", std::move(X));
  SetInputParam("responses", std::move(y));
  SetInputParam("center", 0);

  mlpackMain();

  BayesianRidge* estimator = CLI::GetParam<BayesianRidge*>("output_model");

  const arma::colvec dataScale = estimator->DataScale();
  const arma::colvec dataOffset = estimator->DataOffset();

  BOOST_REQUIRE(sum(dataOffset) == 0);
  BOOST_REQUIRE(sum(dataScale) == m);
}

/**
 * Check predictions of saved model and in code model are equal.
 */
BOOST_AUTO_TEST_CASE(BayesianRidgeSavedEqualCode)
{
  int n = 10, m = 4;
  arma::mat X = arma::randu<arma::mat>(n, m);
  arma::mat Xtest = arma::randu<arma::mat>(2 * n, m);
  const arma::colvec omega = arma::randu<arma::colvec>(m);
  arma::mat y =  X * omega;

  BayesianRidge model;
  model.Train(X.t(), y.t());

  arma::rowvec responses;
  model.Predict(Xtest.t(), responses);

  SetInputParam("input", std::move(X));
  SetInputParam("responses", std::move(y));

  mlpackMain();

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["responses"].wasPassed = false;

  SetInputParam("input_model", CLI::GetParam<BayesianRidge*>("output_model"));
  SetInputParam("test", std::move(Xtest));

  mlpackMain();

  arma::mat ytest = std::move(responses).t();
  // Check that initial output and output using saved model are same.
  CheckMatrices(ytest, CLI::GetParam<arma::mat>("output_predictions"));
}

BOOST_AUTO_TEST_SUITE_END();
