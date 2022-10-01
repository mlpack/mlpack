/**
 * @file krann_test.cpp
 * @author Ryan Curtin
 * @author Utkarsh Rai
 *
 * Test RUN_BINDING() of krann_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BINDING_TYPE BINDING_TYPE_TEST

#include <mlpack/core.hpp>
#include <mlpack/methods/rann/krann_main.cpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "main_test_fixture.hpp"

#include "../catch.hpp"
#include "../test_catch_tools.hpp"

using namespace mlpack;

BINDING_TEST_FIXTURE(KRANNTestFixture);

/*
 * Check that we can't provide reference and query matrices
 * with different dimensions.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNEqualDimensionTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Now we specify an invalid dimension(2) for the query data.
  // Note that the number of points in queryData and referenceData matrices
  // are allowed to be different
  arma::mat queryData;
  queryData.randu(2, 90); // 90 points in 2 dimensions.

  // Random input, some k <= number of top tau percentile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 5);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
 * Check that we can't specify an invalid k when only reference
 * matrix is given.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNInvalidKTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 101);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) -1); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 6); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  // Test on empty reference matrix since referenceData has been moved.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
 * Check that we can't specify an invalid k when both reference
 * and query matrices are given.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNInvalidKQueryDataTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Random input, some k > number of  top tau percentile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 101);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference",  referenceData);
  SetInputParam("k", (int) -1); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 6); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  // Test on empty reference marix since referenceData has been moved.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int)  5);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Check that we can't specify a negative leaf size.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNLeafSizeTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, negative leaf size.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("leaf_size", (int) -1); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
 * Check that we can't pass both input_model and reference matrix.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNRefModelTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of  top tau percentile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);

  RUN_BINDING();

  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(params.Get<RAModel*>("output_model")));

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
 * Check that we can't pass an invalid tree type.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNInvalidTreeTypeTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of  top tau percentile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("tree_type", (string) "min-rp"); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/*
 * Check that we can't pass an invalid value of tau.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNInvalidTauTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of  top tau percentile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("tau", (double) -1); // Invalid.

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}

/**
 * Make sure that dimensions of the neighbors and distances matrices are correct
 * given a value of k.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNOutputDimensionTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of  top tau percentile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);

  FixedRandomSeed();
  RUN_BINDING();

  // Check the neighbors matrix has 5 points for each input point.
  REQUIRE(params.Get<arma::Mat<size_t>>("neighbors").n_rows == 5);
  REQUIRE(params.Get<arma::Mat<size_t>>("neighbors").n_cols == 100);

  // Check the distances matrix has 10 points for each input point.
  REQUIRE(params.Get<arma::mat>("distances").n_rows == 5);
  REQUIRE(params.Get<arma::mat>("distances").n_cols == 100);
}

/**
 * Ensure that saved model can be used again.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNModelReuseTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Random input, some k <= number of  top tau percentile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", queryData);
  SetInputParam("k", (int) 5);

  FixedRandomSeed();
  RUN_BINDING();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  RAModel* output_model;
  neighbors = std::move(params.Get<arma::Mat<size_t>>("neighbors"));
  distances = std::move(params.Get<arma::mat>("distances"));
  output_model = std::move(params.Get<RAModel*>("output_model"));

  // Reset passed parameters.
  params.Get<RAModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", output_model);
  SetInputParam("query", queryData);
  SetInputParam("k", (int) 5);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, params.Get<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, params.Get<arma::mat>("distances"));
}

/**
 * Ensure that different leaf sizes give different results.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNDifferentLeafSizes",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau percentile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);
  SetInputParam("leaf_size", (int) 1);

  FixedRandomSeed();
  RUN_BINDING();

  RAModel* output_model;
  output_model = std::move(params.Get<RAModel*>("output_model"));

  // Reset passed parameters.
  params.Get<RAModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("leaf_size", (int) 10);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CHECK(output_model->LeafSize() == (int) 1);
  CHECK(params.Get<RAModel*>("output_model")->LeafSize() == (int) 10);
  delete output_model;
}

/**
 * Ensure that different tau give different results.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNDifferentTau",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau percentile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  FixedRandomSeed();
  RUN_BINDING();

  RAModel* output_model;
  output_model = std::move(params.Get<RAModel*>("output_model"));

  // Reset the passed parameters.
  params.Get<RAModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Changing value of tau and keeping everything else unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("tau", (double) 10);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal
  CHECK(output_model->Tau() == (double) 5);
  CHECK(params.Get<RAModel*>("output_model")->Tau() ==
      (double) 10);
  delete output_model;
}

/**
 * Ensure that different alpha give different results.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNDifferentAlpha",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau percentile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  FixedRandomSeed();
  RUN_BINDING();

  RAModel* output_model;
  output_model = std::move(params.Get<RAModel*>("output_model"));

  // Reset the passed parameters.
  params.Get<RAModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Changing value of tau and keeping everything else unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("alpha", (double) 0.80);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal
  CHECK(output_model->Alpha() == (double) 0.95);
  CHECK(params.Get<RAModel*>("output_model")->Alpha() ==
      (double) 0.80);
  delete output_model;
}

/**
 * Ensure that different tree-type give different results.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNDifferentTreeType",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau percentile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  FixedRandomSeed();
  RUN_BINDING();

  RAModel* output_model;
  output_model = std::move(params.Get<RAModel*>("output_model"));

  // Reset the passed parameters.
  params.Get<RAModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Changing value of tau and keeping everything else unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("tree_type", (string) "ub");

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal
  const bool check = output_model->TreeType() == 0;
  CHECK(check == true);
  CHECK(params.Get<RAModel*>("output_model")->TreeType() ==
      8);
  delete output_model;
}

/**
 * Ensure that different single_sample_limit gives different results.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNDifferentSingleSampleLimit",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau percentile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  FixedRandomSeed();
  RUN_BINDING();

  RAModel* output_model;
  output_model = std::move(params.Get<RAModel*>("output_model"));

  // Reset passed parameters.
  params.Get<RAModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("single_sample_limit", (int)15);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CHECK(params.Get<RAModel*>("output_model")->SingleSampleLimit() ==
      (int) 15);
  CHECK(output_model->SingleSampleLimit() == (int) 20);
  delete output_model;
}

/**
 * Ensure that toggling sample_at_leaves gives different results.
 */
TEST_CASE_METHOD(KRANNTestFixture, "KRANNDifferentSampleAtLeaves",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau percentile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  FixedRandomSeed();
  RUN_BINDING();

  RAModel* output_model;
  output_model = std::move(params.Get<RAModel*>("output_model"));

  // Reset passed parameters.
  params.Get<RAModel*>("output_model") = NULL;
  CleanMemory();
  ResetSettings();

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("sample_at_leaves", (bool) true);

  FixedRandomSeed();
  RUN_BINDING();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CHECK(params.Get<RAModel*>("output_model")->SampleAtLeaves() ==
      (bool) true);
  CHECK(output_model->SampleAtLeaves() == (bool) false);
  delete output_model;
}

/**
 * Ensure that alpha out of range throws an error.
*/
TEST_CASE_METHOD(KRANNTestFixture, "KRANNInvalidAlphaTest",
                "[KRANNMainTest][BindingTests]")
{
  arma::mat referenceData;
  referenceData.randu(3, 100);

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("alpha", (double) 1.2);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);

  ResetSettings();

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("alpha", (double) -1);

  REQUIRE_THROWS_AS(RUN_BINDING(), std::runtime_error);
}
