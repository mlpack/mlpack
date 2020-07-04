/**
 * @file krann_test.cpp
 * @author Ryan Curtin
 * @author Utkarsh Rai
 *
 * Test mlpackMain() of krann_main.cpp.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license. You should have received a copy of the
 * 3-clause BSD license along with mlpack. If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <string>

#define BINDING_TYPE BINDING_TYPE_TEST
static const std::string testName = "K-RankApproximateNearestNeighborsSearch";

#include <mlpack/core.hpp>
#include <mlpack/core/util/mlpack_main.hpp>
#include "test_helper.hpp"
#include <mlpack/methods/rann/krann_main.cpp>

#include <boost/test/unit_test.hpp>
#include "../test_tools.hpp"

using namespace mlpack;

struct KRANNTestFixture
{
  KRANNTestFixture()
  {
    CLI::RestoreSettings(testName);
  }

  ~KRANNTestFixture()
  {
    bindings::tests::CleanMemory();
    CLI::ClearSettings();
  }
};

BOOST_FIXTURE_TEST_SUITE(KRANNMainTest, KRANNTestFixture);

/*
 * Check that we can't provide reference and query matrices
 * with different dimensions.
 */
BOOST_AUTO_TEST_CASE(KRANNEqualDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Now we specify an invalid dimension(2) for the query data.
  // Note that the number of points in queryData and referenceData matrices
  // are allowed to be different
  arma::mat queryData;
  queryData.randu(2, 90); // 90 points in 2 dimensions.

  // Random input, some k <= number of top tau %ile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 5);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when only reference
 * matrix is given.
 */
BOOST_AUTO_TEST_CASE(KRANNInvalidKTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k > number of reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 101);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) -1); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 6); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  // Test on empty reference matrix since referenceData has been moved.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't specify an invalid k when both reference
 * and query matrices are given.
 */
BOOST_AUTO_TEST_CASE(KRANNInvalidKQueryDataTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Random input, some k > number of  top tau %ile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", std::move(queryData));
  SetInputParam("k", (int) 101);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference",  referenceData);
  SetInputParam("k", (int) -1); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 6); // Invalid.

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["k"].wasPassed = false;

  // Test on empty reference marix since referenceData has been moved.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int)  5);

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Check that we can't specify a negative leaf size.
 */
BOOST_AUTO_TEST_CASE(KRANNLeafSizeTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, negative leaf size.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("leaf_size", (int) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass both input_model and reference matrix.
 */
BOOST_AUTO_TEST_CASE(KRANNRefModelTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of  top tau %ile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);

  mlpackMain();

  // Input pre-trained model.
  SetInputParam("input_model",
      std::move(CLI::GetParam<RANNModel*>("output_model")));

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid tree type.
 */
BOOST_AUTO_TEST_CASE(KRANNInvalidTreeTypeTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of  top tau %ile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("tree_type", (string) "min-rp"); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/*
 * Check that we can't pass an invalid value of tau.
 */
BOOST_AUTO_TEST_CASE(KRANNInvalidTauTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of  top tau %ile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("tau", (double) -1); // Invalid.

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

/**
 * Make sure that dimensions of the neighbors and distances matrices are correct
 * given a value of k.
 */
BOOST_AUTO_TEST_CASE(KRANNOutputDimensionTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of  top tau %ile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check the neighbors matrix has 5 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>("neighbors").n_rows,
      5);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::Mat<size_t>>("neighbors").n_cols,
      100);

  // Check the distances matrix has 10 points for each input point.
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_rows, 5);
  BOOST_REQUIRE_EQUAL(CLI::GetParam<arma::mat>("distances").n_cols, 100);
}

/**
 * Ensure that saved model can be used again.
 */
BOOST_AUTO_TEST_CASE(KRANNModelReuseTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  arma::mat queryData;
  queryData.randu(3, 90); // 90 points in 3 dimensions.

  // Random input, some k <= number of  top tau %ile reference points.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("query", queryData);
  SetInputParam("k", (int) 5);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  arma::Mat<size_t> neighbors;
  arma::mat distances;
  RANNModel* output_model;
  neighbors = std::move(CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  distances = std::move(CLI::GetParam<arma::mat>("distances"));
  output_model = std::move(CLI::GetParam<RANNModel*>("output_model"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["query"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("input_model", output_model);
  SetInputParam("query", queryData);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  CheckMatrices(neighbors, CLI::GetParam<arma::Mat<size_t>>("neighbors"));
  CheckMatrices(distances, CLI::GetParam<arma::mat>("distances"));
}

/**
  * Ensure that different leaf sizes give different results.
 */
BOOST_AUTO_TEST_CASE(KRANNDifferentLeafSizes)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau %ile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);
  SetInputParam("leaf_size", (int) 1);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  RANNModel* output_model;
  output_model = std::move(CLI::GetParam<RANNModel*>("output_model"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("leaf_size", (int) 10);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  BOOST_CHECK_EQUAL(output_model->LeafSize(), (int) 1);
  BOOST_CHECK_EQUAL(CLI::GetParam<RANNModel*>("output_model")->LeafSize(),
      (int) 10);
  delete output_model;
}

/**
 * Ensure that different tau give different results.
 */
BOOST_AUTO_TEST_CASE(KRANNDifferentTau)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau %ile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  RANNModel* output_model;
  output_model = std::move(CLI::GetParam<RANNModel*>("output_model"));

  // Reset the passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Changing value of tau and keeping everything else unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("tau", (double) 10);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal
  BOOST_CHECK_EQUAL(output_model->Tau(), (double) 5);
  BOOST_CHECK_EQUAL(CLI::GetParam<RANNModel*>("output_model")->Tau(),
      (double) 10);
  delete output_model;
}

/**
 * Ensure that different alpha give different results.
 */
BOOST_AUTO_TEST_CASE(KRANNDifferentAlpha)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau %ile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  RANNModel* output_model;
  output_model = std::move(CLI::GetParam<RANNModel*>("output_model"));

  // Reset the passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Changing value of tau and keeping everything else unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("alpha", (double) 0.80);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal
  BOOST_CHECK_EQUAL(output_model->Alpha(), (double) 0.95);
  BOOST_CHECK_EQUAL(CLI::GetParam<RANNModel*>("output_model")->Alpha(),
      (double) 0.80);
  delete output_model;
}

/**
 * Ensure that different tree-type give different results.
 */
BOOST_AUTO_TEST_CASE(KRANNDifferentTreeType)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau %ile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  RANNModel* output_model;
  output_model = std::move(CLI::GetParam<RANNModel*>("output_model"));

  // Reset the passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Changing value of tau and keeping everything else unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("tree_type", (string) "ub");

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal
  BOOST_CHECK_EQUAL(output_model->TreeType(), 0);
  BOOST_CHECK_EQUAL(CLI::GetParam<RANNModel*>("output_model")->TreeType(),
      8);
  delete output_model;
}

/**
  * Ensure that different single_sample_limit gives different results.
 */
BOOST_AUTO_TEST_CASE(KRANNDifferentSingleSampleLimit)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau %ile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  RANNModel* output_model;
  output_model = std::move(CLI::GetParam<RANNModel*>("output_model"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("single_sample_limit", (int)15);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  BOOST_CHECK_EQUAL(
    CLI::GetParam<RANNModel*>("output_model")->SingleSampleLimit(), (int) 15);
  BOOST_CHECK_EQUAL(output_model->SingleSampleLimit(), (int) 20);
  delete output_model;
}

/**
  * Ensure that toggling sample_at_leaves gives different results.
 */
BOOST_AUTO_TEST_CASE(KRANNDifferentSampleAtLeaves)
{
  arma::mat referenceData;
  referenceData.randu(3, 100); // 100 points in 3 dimensions.

  // Random input, some k <= number of top tau %ile reference points.
  SetInputParam("reference", referenceData);
  SetInputParam("k", (int) 5);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  RANNModel* output_model;
  output_model = std::move(CLI::GetParam<RANNModel*>("output_model"));

  // Reset passed parameters.
  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;

  // Input saved model, pass the same query and keep k unchanged.
  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("sample_at_leaves", (bool) true);

  mlpack::math::FixedRandomSeed();
  mlpackMain();

  // Check that initial output matrices and the output matrices using
  // saved model are equal.
  BOOST_CHECK_EQUAL(CLI::GetParam<RANNModel*>("output_model")->SampleAtLeaves(),
      (bool) true);
  BOOST_CHECK_EQUAL(output_model->SampleAtLeaves(), (bool) false);
  delete output_model;
}

/**
  * Ensure that alpha out of range throws an error.
 */
BOOST_AUTO_TEST_CASE(KRANNInvalidAlphaTest)
{
  arma::mat referenceData;
  referenceData.randu(3, 100);

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("k", (int) 5);
  SetInputParam("alpha", (double) 1.2);

  Log::Fatal.ignoreInput = true;
  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);

  CLI::GetSingleton().Parameters()["reference"].wasPassed = false;
  CLI::GetSingleton().Parameters()["alpha"].wasPassed = false;

  SetInputParam("reference", std::move(referenceData));
  SetInputParam("alpha", (double) -1);

  BOOST_REQUIRE_THROW(mlpackMain(), std::runtime_error);
  Log::Fatal.ignoreInput = false;
}

BOOST_AUTO_TEST_SUITE_END();
