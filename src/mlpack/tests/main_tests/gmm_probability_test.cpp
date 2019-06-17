/**
 * @file gmm_probability_test.cpp
 * @author Yashwant Singh
 *
 * Test mlpackMain() of gmm_probability_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#define BINDING_TYPE BINDING_TYPE_TEST

static const std::string testName = "GmmProbability";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/gmm/gmm_probability_main.cpp>

#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>


using namespace mlpack;

struct GmmProbabilityTestFixture
{
 public:
  GmmProbabilityTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~GmmProbabilityTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

void ResetGmmProbabilitySetting()
{
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(GmmProbabilityMainTest, GmmProbabilityTestFixture);

// Checking the input and output dimensionality.
BOOST_AUTO_TEST_CASE(GmmProbabilityDimensionality)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  GMM gmm(1, 5);
  gmm.Train(std::move(inputData), 5);

  arma::mat inputPoints(5, 5, arma::fill::randu);

  SetInputParam("input", std::move(inputPoints));
  SetInputParam("input_model", &gmm);

  mlpackMain();

  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_cols, 5);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("output").n_rows, 1);
}

BOOST_AUTO_TEST_SUITE_END();

