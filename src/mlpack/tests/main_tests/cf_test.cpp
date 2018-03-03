/**
  * @file cf_test.cpp
  * @author Wenhao Huang
  *
  * Test mlpackMain() of cf_main.cpp
  *
  * mlpack is free software; you may redistribute it and/or modify it under the
  * terms of the 3-clause BSD license.  You should have received a copy of the
  * 3-clause BSD license along with mlpack.  If not, see
  * http://www.opensource.org/licenses/BSD-3-Clause for more information.
  */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST

static const std::string testName = "CollaborativeFiltering";

#include <mlpack/core.hpp>
#include <mlpack/methods/cf/cf_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"

#include <boost/test/unit_test.hpp>

using namespace mlpack;
using namespace arma;


struct CFTestFixture
{
 public:
  CFTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~CFTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

static void ResetSettings()
{
  bindings::tests::CleanMemory();
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(CFMainTest, CFTestFixture);

/**
 * Ensure the rank is non-negative.
 */
BOOST_AUTO_TEST_CASE(CFRankBoundTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  // Rank should not be negative.
  SetInputParam("rank", int(-1));
  SetInputParam("training", std::move(dataset));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure min_residue is non-negative.
 */
BOOST_AUTO_TEST_CASE(CFMinResidueBoundTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  // min_residue should not be negative.
  SetInputParam("min_residue", double(-1));
  SetInputParam("training", std::move(dataset));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure max_iterations is positive.
 */
BOOST_AUTO_TEST_CASE(CFMaxIterationsBoundTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  // max_iterations should not be negative.
  SetInputParam("max_iterations", double(-1));
  SetInputParam("training", std::move(dataset));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // max_iterations should not be zero.
  SetInputParam("max_iterations", double(0));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure recommendations is positive.
 */
BOOST_AUTO_TEST_CASE(CFRecommendationsBoundTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  SetInputParam("training", std::move(dataset));
  SetInputParam("max_iterations", int(5));

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["max_iterations"].wasPassed = false;

  // recommendations should not be zero.
  SetInputParam("recommendations", int(0));
  SetInputParam("input_model",
      std::move(CLI::GetParam<CF*>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // recommendations should not be negative.
  SetInputParam("recommendations", int(-1));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure neighborhood is positive.
 */
BOOST_AUTO_TEST_CASE(CFNeighborhoodBoundTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  SetInputParam("training", std::move(dataset));
  SetInputParam("max_iterations", int(5));

  mlpackMain();

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["training"].wasPassed = false;
  CLI::GetSingleton().Parameters()["max_iterations"].wasPassed = false;

  // neighborhood should not be zero.
  SetInputParam("neighborhood", int(0));
  SetInputParam("input_model",
      std::move(CLI::GetParam<CF*>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // neighborhood should not be negative.
  SetInputParam("neighborhood", int(-1));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure algorithm is one of { "NMF", "BatchSVD",
 * "SVDIncompleteIncremental", "SVDCompleteIncremental", "RegSVD" }
 */
BOOST_AUTO_TEST_CASE(CFAlgorithmBoundTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  // algorithm should be valid.
  SetInputParam("algorithm", std::string("invalid_algorithm"));
  SetInputParam("training", std::move(dataset));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}





















BOOST_AUTO_TEST_SUITE_END();
