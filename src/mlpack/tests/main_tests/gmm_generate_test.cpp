/**
 * @file gmm_generate_test.cpp
 * @author Yashwant Singh
 *
 * Test mlpackMain() of gmm_generate_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "GmmGenerate";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include <mlpack/methods/gmm/gmm_generate_main.cpp>

#include "test_helper.hpp"
#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct GmmGenerateTestFixture
{
 public:
  GmmGenerateTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~GmmGenerateTestFixture()
  {
    // Clear the settings.
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(GmmGenerateMainTest, GmmGenerateTestFixture);

// Checking that Samples must greater than 0.
BOOST_AUTO_TEST_CASE(GmmGenerateSamplesTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  GMM gmm(1, 5);
  gmm.Train(inputData, 5);

  SetInputParam("input_model", &gmm);

  Log::Fatal.ignoreInput = true;
  SetInputParam("samples", 0); // Invalid
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Checking dimensionality of output.
BOOST_AUTO_TEST_CASE(GmmGenerateDimensionality)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  GMM gmm(1, 5);
  gmm.Train(inputData, 5);
  SetInputParam("input_model", &gmm);
  SetInputParam("samples", (int) 10);

  mlpackMain();

  arma::mat output = std::move(CLI::GetParam<arma::mat>("output"));

  BOOST_REQUIRE_EQUAL(output.n_rows, gmm.Dimensionality());
  BOOST_REQUIRE_EQUAL(output.n_cols, (int) 10);
}

BOOST_AUTO_TEST_SUITE_END();

