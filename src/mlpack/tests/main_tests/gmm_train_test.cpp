/**
 * @file gmm_train_test.cpp
 * @author Yashwant Singh
 *
 * Test mlpackMain() of gmm_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include<string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "GmmTrain";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/gmm/gmm_train_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct GmmTrainTestFixture
{
  public:
  GmmTrainTestFixture()
  {
    // Cache in the options for this program.
    CLI::RestoreSettings(testName);
  }

  ~GmmTrainTestFixture()
  {
    // Clear the settings.
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

void ResetGmmTrainSetting()
{
  CLI::ClearSettings();
  CLI::RestoreSettings(testName);
}

BOOST_FIXTURE_TEST_SUITE(GmmTrainMainTest, GmmTrainTestFixture);

// To check if the gaussian is positive or not.
BOOST_AUTO_TEST_CASE(GmmTrainValidGaussianTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", 0); // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
  * To check if the number of gaussians in the output model is same as
  * that of input gaussian parameter or not.
 **/
BOOST_AUTO_TEST_CASE(GmmTrainOutputModelGaussianTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("trials", (int) 2);

  mlpackMain();

  GMM* gmm = CLI::GetParam<GMM*>("output_model");
  BOOST_REQUIRE_EQUAL(gmm->Gaussians(), (int) 2);
}

// Max iterations must be positive.
BOOST_AUTO_TEST_CASE(GmmTrainMaxIterationsTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("trials", (int) 1);
  SetInputParam("max_iterations", (int)-1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Ensure that Trials must be greater than 0.
BOOST_AUTO_TEST_CASE(GmmTrainTrialsTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("trials", (int) 0); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Checking that percentage is between 0 and 1.
BOOST_AUTO_TEST_CASE(RefinedStartPercentageTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);

  Log::Fatal.ignoreInput = true;
  SetInputParam("percentage", (double) 2.0); // Invalid
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  SetInputParam("percentage", (double) -1.0); // Invalid
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  Log::Fatal.ignoreInput = false;
}

// Samplings must be positive.
BOOST_AUTO_TEST_CASE(GmmTrainSamplings)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("samplings", (int) 0); // Invalid

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

// Number of gaussians in the model trained from input model.
BOOST_AUTO_TEST_CASE(GmmTrainNumberOfGaussian)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);

  mlpackMain();

  GMM* gmm = CLI::GetParam<GMM*>("output_model");

  SetInputParam("input_model", gmm);

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;

  SetInputParam("input", std::move(inputData));

  mlpackMain();

  GMM* gmm1 = CLI::GetParam<GMM*>("output_model");

  BOOST_REQUIRE_EQUAL(gmm1->Gaussians(), (int) 2);
}

// Ensure that Noise affects the final result.
BOOST_AUTO_TEST_CASE(GmmTrainNoisetest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);
  SetInputParam("noise", (double) 0.0);

  mlpackMain();

  GMM* gmm = CLI::GetParam<GMM*>("output_model");

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["gaussians"].wasPassed = false;
  CLI::GetSingleton().Parameters()["noise"].wasPassed = false;

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("noise", (double) 1.5);
  mlpackMain();

  GMM* gmm1 = CLI::GetParam<GMM*>("output_model");

  arma::uvec sortedIndices = sort_index(gmm->Weights());

  for (size_t k = 0; k < sortedIndices.n_elem; k++)
    CheckMatricesNotEqual(gmm->Component(sortedIndices[k]).Covariance(),
                            gmm1->Component(sortedIndices[k]).Covariance());
}

// Ensure that Percentage affects the final result when refined_start is true.
BOOST_AUTO_TEST_CASE(GmmTrainPercentageTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);
  SetInputParam("refined_start", true);
  SetInputParam("percentage", (double) 0.02);

  mlpackMain();

  GMM* gmm = CLI::GetParam<GMM*>("output_model");

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["gaussians"].wasPassed = false;
  CLI::GetSingleton().Parameters()["refined_start"].wasPassed = false;
  CLI::GetSingleton().Parameters()["percentage"].wasPassed = false;

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("refined_start", true);
  SetInputParam("percentage", (double) 0.52);

  mlpackMain();

  GMM* gmm1 = CLI::GetParam<GMM*>("output_model");

  arma::uvec sortedIndices = sort_index(gmm->Weights());

  for (size_t k = 0; k < sortedIndices.n_elem; k++)
    CheckMatricesNotEqual(gmm->Component(sortedIndices[k]).Covariance(),
                            gmm1->Component(sortedIndices[k]).Covariance());
}

// Ensure that Sampling affects the final result when refined_start is true.
BOOST_AUTO_TEST_CASE(GmmTrainSamplingsTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);
  SetInputParam("refined_start", true);
  SetInputParam("percentage", (double) 0.5);
  SetInputParam("samplings", (int) 100);

  mlpackMain();

  GMM* gmm = CLI::GetParam<GMM*>("output_model");

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["gaussians"].wasPassed = false;
  CLI::GetSingleton().Parameters()["refined_start"].wasPassed = false;
  CLI::GetSingleton().Parameters()["percentage"].wasPassed = false;
  CLI::GetSingleton().Parameters()["samplings"].wasPassed = false;

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("refined_start", true);
  SetInputParam("percentage", (double) 0.5);
  SetInputParam("samplings", (int) 500);

  mlpackMain();

  GMM* gmm1 = CLI::GetParam<GMM*>("output_model");

  arma::uvec sortedIndices = sort_index(gmm->Weights());

  for (size_t k = 0; k < sortedIndices.n_elem; k++)
    CheckMatricesNotEqual(gmm->Component(sortedIndices[k]).Covariance(),
                            gmm1->Component(sortedIndices[k]).Covariance());

}

// Ensure that tolerance affects the final result.
BOOST_AUTO_TEST_CASE(GmmTrainToleranceTest)
{
  arma::mat inputData;
  if (!data::Load("vc2.csv", inputData))
    BOOST_FAIL("Unable to load train dataset vc2.csv!");

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);
  SetInputParam("tolerance", (double) 1e-10);

  mlpackMain();

  GMM* gmm = CLI::GetParam<GMM*>("output_model");

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;
  CLI::GetSingleton().Parameters()["gaussians"].wasPassed = false;
  CLI::GetSingleton().Parameters()["tolerance"].wasPassed = false;

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("tolerance", (double) 1e-30);

  mlpackMain();

  GMM* gmm1 = CLI::GetParam<GMM*>("output_model");

  arma::uvec sortedIndices = sort_index(gmm->Weights());

  for (size_t k = 0; k < sortedIndices.n_elem; k++)
    CheckMatricesNotEqual(gmm->Component(sortedIndices[k]).Covariance(),
                            gmm1->Component(sortedIndices[k]).Covariance());
}

// Ensure that saved model can be used again.
BOOST_AUTO_TEST_CASE(GmmTrainModelReuseTest)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);

  mlpackMain();

  GMM* gmm = CLI::GetParam<GMM*>("output_model");

  SetInputParam("input_model", gmm);

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;

  SetInputParam("input", inputData);

  mlpackMain();

  GMM* gmm1 = CLI::GetParam<GMM*>("output_model");

  SetInputParam("input_model", gmm1);

  CLI::GetSingleton().Parameters()["input"].wasPassed = false;

  SetInputParam("input", std::move(inputData));

  mlpackMain();

  GMM* gmm2 = CLI::GetParam<GMM*>("output_model");

  BOOST_REQUIRE_EQUAL(gmm1, gmm2);
}

// Ensure that Gmm's covariances are diagonal when diagonal_covariance is true.
BOOST_AUTO_TEST_CASE(GmmTrainDiagCovariance)
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("diagonal_covariance", true);

  mlpackMain();

  GMM* gmm = CLI::GetParam<GMM*>("output_model");

  arma::uvec sortedIndices = sort_index(gmm->Weights());

  for (size_t k = 0; k < sortedIndices.n_elem; k++)
  {
    arma::mat diagCov(gmm->Component(sortedIndices[k]).Covariance());
      for (size_t i = 0; i < diagCov.n_rows; i++)
        for (size_t j = 0; j < diagCov.n_cols; j++)
          if (i != j && diagCov(i, j) != (double) 0)
            BOOST_FAIL("Covariance Are Not Diagonal");
  }
}

BOOST_AUTO_TEST_SUITE_END();

