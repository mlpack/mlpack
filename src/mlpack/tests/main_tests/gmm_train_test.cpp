/**
 * @file tests/main_tests/gmm_train_test.cpp
 * @author Yashwant Singh
 *
 * Test RUN_BINDING() of gmm_train_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/gmm/gmm_train_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(GmmTrainTestFixture);

inline bool CheckDifferent(GMM* gmm1, GMM* gmm2)
{
  bool different = (arma::norm(gmm1->Weights() - gmm2->Weights()) > 1e-50);
  if (!different)
  {
    for (size_t i = 0; i < gmm1->Weights().n_elem; ++i)
    {
      if (arma::norm(gmm1->Component(i).Mean() -
                     gmm2->Component(i).Mean()) > 1e-50)
      {
        different = true;
        break;
      }

      if (arma::norm(gmm1->Component(i).Covariance() -
                     gmm2->Component(i).Covariance()) > 1e-50)
      {
        different = true;
        break;
      }
    }
  }

  return different;
}

// To check if the gaussian is positive or not.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainValidGaussianTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", 0); // Invalid

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
  * To check if the number of gaussians in the output model is same as
  * that of input gaussian parameter or not.
 **/
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainOutputModelGaussianTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("trials", (int) 2);

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");
  REQUIRE(gmm->Gaussians() == (int) 2);
}

// Max iterations must be positive.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainMaxIterationsTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("trials", (int) 1);
  SetInputParam("max_iterations", (int)-1); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Ensure that Trials must be greater than 0.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainPositiveTrialsTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("trials", (int) 0); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Checking that percentage is between 0 and 1.
TEST_CASE_METHOD(GmmTrainTestFixture, "GMMRefinedStartPercentageTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);

  SetInputParam("percentage", (double) 2.0); // Invalid
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  SetInputParam("percentage", (double) -1.0); // Invalid
  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

}

// Samplings must be positive.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainSamplings",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("refined_start", true);
  SetInputParam("samplings", (int) 0); // Invalid

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

// Number of gaussians in the model trained from input model.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainNumberOfGaussian",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");

  ResetSettings();

  SetInputParam("input_model", gmm);
  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);

  RUN_BINDING();

  GMM* gmm1 = params.Get<GMM*>("output_model");

  REQUIRE(gmm1->Gaussians() == (int) 2);
}

// Making sure that enabling no_force_positive doesn't crash.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainNoForcePositiveTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 1);
  SetInputParam("no_force_positive", true);

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");

  ResetSettings();

  SetInputParam("input_model", gmm);
  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 1);
  SetInputParam("no_force_positive", true);

  RUN_BINDING();

  GMM* gmm1 = params.Get<GMM*>("output_model");

  REQUIRE(gmm1->Gaussians() == (int) 1);
}

// Ensure that Noise affects the final result.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainNoiseTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("data_3d_mixed.txt", inputData))
    FAIL("Unable to load train dataset data_3d_mixed.txt!");

  FixedRandomSeed();

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);
  SetInputParam("noise", (double) 0.0);

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");
  params.Get<GMM*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("noise", (double) 100.0);

  FixedRandomSeed();

  RUN_BINDING();

  GMM* gmm1 = params.Get<GMM*>("output_model");

  REQUIRE(CheckDifferent(gmm, gmm1));

  delete gmm;
}

// Ensure that Trials affects the final result.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainTrialsTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(10, 1000, arma::fill::randu);

  // We don't require that this passes every time, since it is possible that the
  // end result can be an identical model.  Instead, we only require that it's
  // different at least one in ten times, because that means the "trials" option
  // is making a difference.
  bool success = false;
  for (size_t trial = 0; trial < 10; ++trial)
  {
    CustomRandomSeed(trial);

    SetInputParam("input", inputData);
    SetInputParam("gaussians", (int) 5);
    SetInputParam("trials", (int) 1);
    SetInputParam("max_iterations", (int) 1);
    SetInputParam("kmeans_max_iterations", (int) 1);

    RUN_BINDING();

    GMM* gmm = params.Get<GMM*>("output_model");
    params.Get<GMM*>("output_model") = NULL;

    CleanMemory();
    ResetSettings();

    SetInputParam("input", inputData);
    SetInputParam("gaussians", (int) 5);
    SetInputParam("trials", (int) 100);
    SetInputParam("max_iterations", (int) 1);
    SetInputParam("kmeans_max_iterations", (int) 1);

    CustomRandomSeed(trial);

    RUN_BINDING();

    GMM* gmm1 = params.Get<GMM*>("output_model");

    success = CheckDifferent(gmm, gmm1);

    delete gmm;

    if (success)
      break;

    CleanMemory();
  }

  REQUIRE(success == true);
}

// Ensure that the maximum number of iterations affects the result.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainDiffMaxIterationsTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 150, arma::fill::randu);

  FixedRandomSeed();

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 3);
  SetInputParam("trials", (int) 1);
  SetInputParam("max_iterations", (int) 1);
  SetInputParam("kmeans_max_iterations", (int) 1);

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");
  params.Get<GMM*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 3);
  SetInputParam("trials", (int) 1);
  SetInputParam("max_iterations", (int) 1000);
  SetInputParam("kmeans_max_iterations", (int) 1);

  FixedRandomSeed();

  RUN_BINDING();

  GMM* gmm1 = params.Get<GMM*>("output_model");

  REQUIRE(CheckDifferent(gmm, gmm1));

  delete gmm;
}

// Ensure that the maximum number of k-means iterations affects the result.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainDiffKmeansMaxIterationsTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 150, arma::fill::randu);

  // We don't require that this passes every time, since it is possible that the
  // end result can be an identical model.  Instead, we only require that it's
  // different at least one in ten times, because that means the "trials" option
  // is making a difference.
  bool success = false;
  for (size_t trial = 0; trial < 10; ++trial)
  {
    CustomRandomSeed(trial);

    SetInputParam("input", inputData);
    SetInputParam("gaussians", (int) 3);
    SetInputParam("trials", (int) 1);
    SetInputParam("max_iterations", (int) 1);
    SetInputParam("kmeans_max_iterations", (int) 1);

    RUN_BINDING();

    GMM* gmm = params.Get<GMM*>("output_model");
    params.Get<GMM*>("output_model") = NULL;

    CleanMemory();
    ResetSettings();

    SetInputParam("input", std::move(inputData));
    SetInputParam("gaussians", (int) 3);
    SetInputParam("trials", (int) 1);
    SetInputParam("max_iterations", (int) 1);
    SetInputParam("kmeans_max_iterations", (int) 1000);

    CustomRandomSeed(trial);

    RUN_BINDING();

    GMM* gmm1 = params.Get<GMM*>("output_model");
    params.Get<GMM*>("output_model") = NULL;

    CleanMemory();
    ResetSettings();

    success = CheckDifferent(gmm, gmm1);

    delete gmm;
    delete gmm1;

    if (success)
      break;

    CleanMemory();
  }

  REQUIRE(success == true);
}

// Ensure that Percentage affects the final result when refined_start is true.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainPercentageTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("data_3d_mixed.txt", inputData))
    FAIL("Unable to load train dataset data_3d_mixed.txt!");

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);
  SetInputParam("refined_start", true);
  SetInputParam("percentage", (double) 0.01);
  SetInputParam("samplings", (int) 1000);

  FixedRandomSeed();

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");
  params.Get<GMM*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("refined_start", true);
  SetInputParam("percentage", (double) 0.45);
  SetInputParam("samplings", (int) 1000);

  FixedRandomSeed();

  RUN_BINDING();

  GMM* gmm1 = params.Get<GMM*>("output_model");

  REQUIRE(CheckDifferent(gmm, gmm1));

  delete gmm;
}

// Ensure that Sampling affects the final result when refined_start is true.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainSamplingsTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("data_3d_mixed.txt", inputData))
    FAIL("Unable to load train dataset data_3d_mixed.txt!");

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 8);
  SetInputParam("refined_start", true);
  SetInputParam("trials", (int) 2);
  SetInputParam("samplings", (int) 10);

  FixedRandomSeed();

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");
  params.Get<GMM*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 8);
  SetInputParam("refined_start", true);
  SetInputParam("trials", (int) 2);
  SetInputParam("samplings", (int) 5000);

  FixedRandomSeed();

  RUN_BINDING();

  GMM* gmm1 = params.Get<GMM*>("output_model");

  REQUIRE(CheckDifferent(gmm, gmm1));

  delete gmm;
}

// Ensure that tolerance affects the final result.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainToleranceTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData;
  if (!data::Load("data_3d_mixed.txt", inputData))
    FAIL("Unable to load train dataset data_3d_mixed.txt!");

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);
  SetInputParam("tolerance", (double) 1e-8);

  FixedRandomSeed();

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");
  params.Get<GMM*>("output_model") = NULL;

  CleanMemory();
  ResetSettings();

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("tolerance", (double) 10);

  FixedRandomSeed();

  RUN_BINDING();

  GMM* gmm1 = params.Get<GMM*>("output_model");

  REQUIRE(CheckDifferent(gmm, gmm1));

  delete gmm;
}

// Ensure that saved model can be used again.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainModelReuseTest",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");

  ResetSettings();

  SetInputParam("input_model", gmm);
  SetInputParam("input", inputData);
  SetInputParam("gaussians", (int) 2);

  RUN_BINDING();

  GMM* gmm1 = params.Get<GMM*>("output_model");

  ResetSettings();

  SetInputParam("input_model", gmm1);
  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);

  RUN_BINDING();

  GMM* gmm2 = params.Get<GMM*>("output_model");

  REQUIRE(gmm1 == gmm2);
}

// Ensure that Gmm's covariances are diagonal when diagonal_covariance is true.
TEST_CASE_METHOD(GmmTrainTestFixture, "GmmTrainDiagCovariance",
                 "[GmmTrainMainTest][BindingTests]")
{
  arma::mat inputData(5, 10, arma::fill::randu);

  SetInputParam("input", std::move(inputData));
  SetInputParam("gaussians", (int) 2);
  SetInputParam("diagonal_covariance", true);

  RUN_BINDING();

  GMM* gmm = params.Get<GMM*>("output_model");

  arma::uvec sortedIndices = sort_index(gmm->Weights());

  for (size_t k = 0; k < sortedIndices.n_elem; ++k)
  {
    arma::mat diagCov(gmm->Component(sortedIndices[k]).Covariance());
      for (size_t i = 0; i < diagCov.n_rows; ++i)
        for (size_t j = 0; j < diagCov.n_cols; ++j)
          if (i != j && diagCov(i, j) != (double) 0)
            FAIL("Covariance is not diagonal");
  }
}
