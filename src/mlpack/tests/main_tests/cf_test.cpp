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
#include <mlpack/core/math/random.hpp>
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
 * Ensure max_iterations is non-negative.
 */
BOOST_AUTO_TEST_CASE(CFMaxIterationsBoundTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  // max_iterations should not be negative.
  SetInputParam("max_iterations", int(-1));
  SetInputParam("training", std::move(dataset));

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

  // recommendations should not be zero.
  SetInputParam("recommendations", int(0));
  SetInputParam("all_user_recommendations", true);
  SetInputParam("training", std::move(dataset));
  SetInputParam("max_iterations", int(5));

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
 * Ensure neighborhood is positive and not larger than the number of users.
 */
BOOST_AUTO_TEST_CASE(CFNeighborhoodBoundTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);
  const size_t userNum = max(dataset.row(0)) + 1;

  // neighborhood should not be zero.
  SetInputParam("neighborhood", int(0));
  SetInputParam("training", std::move(dataset));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // neighborhood should not be negative.
  SetInputParam("neighborhood", int(-1));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;

  // neighborhood should not be larger than the number of users.
  SetInputParam("neighborhood", int(userNum + 1));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Ensure algorithm is one of { "NMF", "BatchSVD",
 * "SVDIncompleteIncremental", "SVDCompleteIncremental", "RegSVD" }.
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

/**
 * Ensure saved models can be reused again.
 */
BOOST_AUTO_TEST_CASE(CFModelReuseTest)
{
  const size_t algorithmNum = 5;
  std::string algorithms[algorithmNum] = { "NMF", "BatchSVD",
      "SVDIncompleteIncremental", "SVDCompleteIncremental", "RegSVD" };

  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  for (size_t i = 0; i < algorithmNum; i++)
  {
    ResetSettings();
    SetInputParam("training", dataset);
    SetInputParam("max_iterations", int(10));
    SetInputParam("algorithm", algorithms[i]);

    mlpackMain();

    // Reset passed parameters.
    CLI::GetSingleton().Parameters()["training"].wasPassed = false;
    CLI::GetSingleton().Parameters()["max_iterations"].wasPassed = false;
    CLI::GetSingleton().Parameters()["algorithm"].wasPassed = false;

    // Reuse the model to get recommendations.
    int recommendations = 3;
    const int querySize = 7;
    Mat<size_t> query =
        arma::linspace<Mat<size_t>>(0, querySize - 1, querySize);

    SetInputParam("query", std::move(query));
    SetInputParam("recommendations", recommendations);
    SetInputParam("input_model",
        std::move(CLI::GetParam<CF*>("output_model")));

    mlpackMain();

    const Mat<size_t>& output = CLI::GetParam<Mat<size_t>>("output");

    BOOST_REQUIRE_EQUAL(output.n_rows, recommendations);
    BOOST_REQUIRE_EQUAL(output.n_cols, querySize);
  }
}

/**
 * Ensure output of all_user_recommendations has correct size.
 */
BOOST_AUTO_TEST_CASE(CFAllUserRecommendationsTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);
  const size_t userNum = max(dataset.row(0)) + 1;

  SetInputParam("training", std::move(dataset));
  SetInputParam("max_iterations", int(10));
  SetInputParam("all_user_recommendations", true);

  mlpackMain();

  const Mat<size_t>& output = CLI::GetParam<Mat<size_t>>("output");

  BOOST_REQUIRE_EQUAL(output.n_cols, userNum);
}

/**
 * Test that rank is used.
 */
BOOST_AUTO_TEST_CASE(CFRankTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);
  int rank = 7;

  SetInputParam("training", std::move(dataset));
  SetInputParam("rank", rank);
  SetInputParam("max_iterations", int(10));
  SetInputParam("algorithm", std::string("NMF"));

  mlpackMain();

  const CF* outputModel = CLI::GetParam<CF*>("output_model");

  BOOST_REQUIRE_EQUAL(outputModel->Rank(), rank);
}

/**
 * Test that min_residue is used.
 */
BOOST_AUTO_TEST_CASE(CFMinResidueTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);
  const CF* outputModel;

  // Set a larger min_residue.
  SetInputParam("min_residue", double(100));
  SetInputParam("training", dataset);
  // Remove the influence of max_iterations.
  SetInputParam("max_iterations", int(1e4));

  // The execution of CF algorithm depends on initial random seed.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  outputModel =  CLI::GetParam<CF*>("output_model");
  const mat w1 = outputModel->W();
  const mat h1 = outputModel->H();

  ResetSettings();

  // Set a smaller min_residue.
  SetInputParam("min_residue", double(0.1));
  SetInputParam("training", std::move(dataset));
  // Remove the influence of max_iterations.
  SetInputParam("max_iterations", int(1e4));

  // The execution of CF algorithm depends on initial random seed.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  outputModel = CLI::GetParam<CF*>("output_model");
  const mat w2 = outputModel->W();
  const mat h2 = outputModel->H();

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::norm(w1 - w2) > 1e-5 || arma::norm(h1 - h2) > 1e-5);
}

/**
 * Test that itertaion_only_termination is used.
 */
BOOST_AUTO_TEST_CASE(CFIterationOnlyTerminationTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);
  const CF* outputModel;

  // Set iteration_only_termination.
  SetInputParam("iteration_only_termination", true);
  SetInputParam("training", dataset);
  SetInputParam("max_iterations", int(100));
  SetInputParam("min_residue", double(1e9));

  // The execution of CF algorithm depends on initial random seed.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  outputModel =  CLI::GetParam<CF*>("output_model");
  const mat w1 = outputModel->W();
  const mat h1 = outputModel->H();

  ResetSettings();

  // Do not set iteration_only_termination.
  SetInputParam("training", std::move(dataset));
  SetInputParam("max_iterations", int(100));
  SetInputParam("min_residue", double(1e9));

  // The execution of CF algorithm depends on initial random seed.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  outputModel = CLI::GetParam<CF*>("output_model");
  const mat w2 = outputModel->W();
  const mat h2 = outputModel->H();

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::norm(w1 - w2) > 1e-5 || arma::norm(h1 - h2) > 1e-5);
}

/**
 * Test that max_iterations is used.
 */
BOOST_AUTO_TEST_CASE(CFMaxIterationsTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);
  const CF* outputModel;

  // Set a larger max_iterations.
  SetInputParam("max_iterations", int(100));
  SetInputParam("training", dataset);
  SetInputParam("iteration_only_termination", true);

  // The execution of CF algorithm depends on initial random seed.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  outputModel =  CLI::GetParam<CF*>("output_model");
  const mat w1 = outputModel->W();
  const mat h1 = outputModel->H();

  ResetSettings();

  // Set a smaller max_iterations.
  SetInputParam("max_iterations", int(5));
  SetInputParam("training", std::move(dataset));
  SetInputParam("iteration_only_termination", true);

  // The execution of CF algorithm depends on initial random seed.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  outputModel = CLI::GetParam<CF*>("output_model");
  const mat w2 = outputModel->W();
  const mat h2 = outputModel->H();

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::norm(w1 - w2) > 1e-5 || arma::norm(h1 - h2) > 1e-5);
}

/**
 * Test that neighborhood is used.
 */
BOOST_AUTO_TEST_CASE(CFNeighborhoodTest)
{
  mat dataset;
  data::Load("GroupLensSmall.csv", dataset);

  const int querySize = 7;
  Mat<size_t> query = arma::linspace<Mat<size_t>>(0, querySize - 1, querySize);

  SetInputParam("neighborhood", int(1));
  SetInputParam("training", dataset);
  SetInputParam("max_iterations", int(10));
  SetInputParam("query", query);

  // The execution of CF algorithm depends on initial random seed.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  const arma::Mat<size_t> output1 = CLI::GetParam<arma::Mat<size_t>>("output");

  ResetSettings();

  // Set a different value for neighborhood.
  SetInputParam("neighborhood", int(10));
  SetInputParam("training", std::move(dataset));
  SetInputParam("max_iterations", int(10));
  SetInputParam("query", std::move(query));

  // The execution of CF algorithm depends on initial random seed.
  mlpack::math::FixedRandomSeed();
  mlpackMain();

  const arma::Mat<size_t> output2 = CLI::GetParam<arma::Mat<size_t>>("output");

  // The resulting matrices should be different.
  BOOST_REQUIRE(arma::any(arma::vectorise(output1 != output2)));
}

BOOST_AUTO_TEST_SUITE_END();
