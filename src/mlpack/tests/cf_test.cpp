/**
 * @file tests/cf_test.cpp
 * @author Mudit Raj Gupta
 * @author Haritha Nair
 *
 * Test file for CF class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/cf.hpp>

#include "catch.hpp"
#include "test_catch_tools.hpp"
#include "serialization.hpp"

using namespace mlpack;
using namespace std;

// Get train and test datasets.
static void GetDatasets(arma::mat& dataset, arma::mat& savedCols)
{
  if (!data::Load("GroupLensSmall.csv", dataset))
    FAIL("Cannot load test dataset GroupLensSmall.csv!");
  savedCols.set_size(3, 50);

  // Save the columns we've removed.
  savedCols.fill(/* random very large value */ 10000000);
  size_t currentCol = 0;
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (currentCol == 50)
      break;

    if (dataset(2, i) > 4.5) // 5-star rating.
    {
      // Make sure we don't have this user yet.  This is a slow way to do this
      // but I don't particularly care here because it's in the tests.
      bool found = false;
      for (size_t j = 0; j < currentCol; ++j)
      {
        if (savedCols(0, j) == dataset(0, i))
        {
          found = true;
          break;
        }
      }

      // If this user doesn't already exist in savedCols, add them.
      // Otherwise ignore this point.
      if (!found)
      {
        savedCols.col(currentCol) = dataset.col(i);
        dataset.shed_col(i);
        ++currentCol;
      }
    }
  }
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set. Default case.
 */
template<typename DecompositionPolicy>
void GetRecommendationsAllUsers()
{
  DecompositionPolicy decomposition;
  // Dummy number of recommendations.
  size_t numRecs = 3;
  // GroupLensSmall.csv dataset has 200 users.
  size_t numUsers = 200;

  // Matrix to save recommendations into.
  arma::Mat<size_t> recommendations;

  // Load GroupLens data.
  arma::mat dataset;
  if (!data::Load("GroupLensSmall.csv", dataset))
    FAIL("Cannot load test dataset GroupLensSamll.csv!");

  CFType<DecompositionPolicy> c(dataset, decomposition, 5, 5, 30);

  // Generate recommendations when query set is not specified.
  c.GetRecommendations(numRecs, recommendations);

  // Check if correct number of recommendations are generated.
  REQUIRE(recommendations.n_rows == numRecs);

  // Check if recommendations are generated for all users.
  REQUIRE(recommendations.n_cols == numUsers);
}

/**
 * Make sure that the recommendations are generated for queried users only.
 */
template<typename DecompositionPolicy>
void GetRecommendationsQueriedUser()
{
  DecompositionPolicy decomposition;
  // Number of users that we will search for recommendations for.
  size_t numUsers = 10;

  // Default number of recommendations.
  size_t numRecsDefault = 5;

  // Create dummy query set.
  arma::Col<size_t> users = arma::zeros<arma::Col<size_t> >(numUsers, 1);
  for (size_t i = 0; i < numUsers; ++i)
    users(i) = i;

  // Matrix to save recommendations into.
  arma::Mat<size_t> recommendations;

  // Load GroupLens data.
  arma::mat dataset;
  if (!data::Load("GroupLensSmall.csv", dataset))
    FAIL("Cannot load test dataset GroupLensSmall.csv!");

  CFType<DecompositionPolicy> c(dataset, decomposition, 5, 5, 30);

  // Generate recommendations when query set is specified.
  c.GetRecommendations(numRecsDefault, recommendations, users);

  // Check if correct number of recommendations are generated.
  REQUIRE(recommendations.n_rows == numRecsDefault);

  // Check if recommendations are generated for the right number of users.
  REQUIRE(recommendations.n_cols == numUsers);
}

/**
 * Make sure recommendations that are generated are reasonably accurate.
 */
template<typename DecompositionPolicy,
         typename NormalizationType = NoNormalization>
void RecommendationAccuracy(const size_t allowedFailures = 17)
{
  DecompositionPolicy decomposition;

  // Small GroupLens dataset.
  arma::mat dataset;

  // Save the columns we've removed.
  arma::mat savedCols;

  GetDatasets(dataset, savedCols);

  CFType<DecompositionPolicy,
      NormalizationType> c(dataset, decomposition, 5, 5, 30);

  // Obtain 150 recommendations for the users in savedCols, and make sure the
  // missing item shows up in most of them.  First, create the list of users,
  // which requires casting from doubles...
  arma::Col<size_t> users(50);
  for (size_t i = 0; i < 50; ++i)
    users(i) = (size_t) savedCols(0, i);
  arma::Mat<size_t> recommendations;
  size_t numRecs = 150;
  c.GetRecommendations(numRecs, recommendations, users);

  REQUIRE(recommendations.n_rows == numRecs);
  REQUIRE(recommendations.n_cols == 50);

  size_t failures = 0;
  for (size_t i = 0; i < 50; ++i)
  {
    size_t targetItem = (size_t) savedCols(1, i);
    bool found = false;
    // Make sure the target item shows up in the recommendations.
    for (size_t j = 0; j < numRecs; ++j)
    {
      const size_t user = users(i);
      const size_t item = recommendations(j, i);
      if (item == targetItem)
      {
        found = true;
      }
      else
      {
        // Make sure we aren't being recommended an item that the user already
        // rated.
        REQUIRE((double) c.CleanedData()(item, user) == 0.0);
      }
    }

    if (!found)
      ++failures;
  }

  // Make sure the right item showed up in at least 2/3 of the recommendations.
  REQUIRE(failures < allowedFailures);
}

// Make sure that Predict() is returning reasonable results.
template<typename DecompositionPolicy,
         typename NormalizationType = OverallMeanNormalization,
         typename NeighborSearchPolicy = EuclideanSearch,
         typename InterpolationPolicy = AverageInterpolation>
void CFPredict(const double rmseBound = 1.5)
{
  // We run the test multiple times, since it sometimes fails, in order to get
  // the probability of failure down.
  bool success = false;
  const size_t trials = 8;
  for (size_t trial = 0; trial < trials; ++trial)
  {
    DecompositionPolicy decomposition;

    // Small GroupLens dataset.
    arma::mat dataset;

    // Save the columns we've removed.
    arma::mat savedCols;

    GetDatasets(dataset, savedCols);

    CFType<DecompositionPolicy,
        NormalizationType> c(dataset, decomposition, 5, 5, 30);

    // Now, for each removed rating, make sure the prediction is... reasonably
    // accurate.
    double totalError = 0.0;
    for (size_t i = 0; i < savedCols.n_cols; ++i)
    {
      const double prediction = c.template Predict<NeighborSearchPolicy,
          InterpolationPolicy>(savedCols(0, i), savedCols(1, i));

      const double error = std::pow(prediction - savedCols(2, i), 2.0);
      totalError += error;
    }

    const double rmse = std::sqrt(totalError / savedCols.n_cols);
    if (rmse < rmseBound)
    {
      success = true;
      break;
    }
  }

  REQUIRE(success == true);
}

// Do the same thing as the previous test, but ensure that the ratings we
// predict with the batch Predict() are the same as the individual Predict()
// calls.
template<typename DecompositionPolicy>
void BatchPredict()
{
  DecompositionPolicy decomposition;

  // Small GroupLens dataset.
  arma::mat dataset;

  // Save the columns we've removed.
  arma::mat savedCols;

  GetDatasets(dataset, savedCols);

  CFType<DecompositionPolicy> c(dataset, decomposition, 5, 5, 30);

  // Get predictions for all user/item pairs we held back.
  arma::Mat<size_t> combinations(2, savedCols.n_cols);
  for (size_t i = 0; i < savedCols.n_cols; ++i)
  {
    combinations(0, i) = size_t(savedCols(0, i));
    combinations(1, i) = size_t(savedCols(1, i));
  }

  arma::vec predictions;
  c.Predict(combinations, predictions);

  for (size_t i = 0; i < combinations.n_cols; ++i)
  {
    const double prediction = c.Predict(combinations(0, i), combinations(1, i));
    REQUIRE(prediction == Approx(predictions[i]).epsilon(1e-10));
  }
}

/**
 * Make sure we can train an already-trained model and it works okay.
 */
template<typename DecompositionPolicy>
void Train(DecompositionPolicy& decomposition)
{
  // Generate random data.
  arma::sp_mat randomData;
  randomData.sprandu(100, 100, 0.3);
  CFType<DecompositionPolicy> c(randomData, decomposition, 5, 5, 30);

  // Small GroupLens dataset.
  arma::mat dataset;

  // Save the columns we've removed.
  arma::mat savedCols;

  GetDatasets(dataset, savedCols);

  // Make data into sparse matrix.
  arma::sp_mat cleanedData;
  CFType<DecompositionPolicy>::CleanData(dataset, cleanedData);

  // Now retrain.
  c.Train(dataset, decomposition, 30);

  // Get predictions for all user/item pairs we held back.
  arma::Mat<size_t> combinations(2, savedCols.n_cols);
  for (size_t i = 0; i < savedCols.n_cols; ++i)
  {
    combinations(0, i) = size_t(savedCols(0, i));
    combinations(1, i) = size_t(savedCols(1, i));
  }

  arma::vec predictions;
  c.Predict(combinations, predictions);

  for (size_t i = 0; i < combinations.n_cols; ++i)
  {
    const double prediction = c.Predict(combinations(0, i),
      combinations(1, i));
    REQUIRE(prediction == Approx(predictions[i]).epsilon(1e-10));
  }
}

/**
 * Make sure we can train an already-trained model and it works okay
 * for policies that use coordinate lists.
 */
template<typename DecompositionPolicy>
void TrainWithCoordinateList(DecompositionPolicy& decomposition)
{
  arma::mat randomData(3, 100);
  randomData.row(0) = arma::linspace<arma::rowvec>(0, 99, 100);
  randomData.row(1) = randomData.row(0);
  randomData.row(2).fill(3);
  CFType<DecompositionPolicy> c(randomData, decomposition, 5, 5, 30);

  // Now retrain with data we know about.
  // Small GroupLens dataset.
  arma::mat dataset;

  // Save the columns we've removed.
  arma::mat savedCols;

  GetDatasets(dataset, savedCols);

  // Now retrain.
  c.Train(dataset, decomposition, 30);

  // Get predictions for all user/item pairs we held back.
  arma::Mat<size_t> combinations(2, savedCols.n_cols);
  for (size_t i = 0; i < savedCols.n_cols; ++i)
  {
    combinations(0, i) = size_t(savedCols(0, i));
    combinations(1, i) = size_t(savedCols(1, i));
  }

  arma::vec predictions;
  c.Predict(combinations, predictions);

  for (size_t i = 0; i < combinations.n_cols; ++i)
  {
    const double prediction = c.Predict(combinations(0, i), combinations(1, i));
    REQUIRE(prediction == Approx(predictions[i]).epsilon(1e-10));
  }
}

/**
 * Make sure we can train a model after using the empty constructor.
 */
template<typename DecompositionPolicy>
void EmptyConstructorTrain()
{
  DecompositionPolicy decomposition;
  // Use default constructor.
  CFType<DecompositionPolicy> c;

  // Now retrain with data we know about.
  // Small GroupLens dataset.
  arma::mat dataset;

  // Save the columns we've removed.
  arma::mat savedCols;

  GetDatasets(dataset, savedCols);

  c.Train(dataset, decomposition, 30);

  // Get predictions for all user/item pairs we held back.
  arma::Mat<size_t> combinations(2, savedCols.n_cols);
  for (size_t i = 0; i < savedCols.n_cols; ++i)
  {
    combinations(0, i) = size_t(savedCols(0, i));
    combinations(1, i) = size_t(savedCols(1, i));
  }

  arma::vec predictions;
  c.Predict(combinations, predictions);

  for (size_t i = 0; i < combinations.n_cols; ++i)
  {
    const double prediction = c.Predict(combinations(0, i),
        combinations(1, i));
    REQUIRE(prediction == Approx(predictions[i]).epsilon(1e-10));
  }
}

/**
 * Ensure we can load and save the CF model.
 */
template<typename DecompositionPolicy,
         typename NormalizationType = NoNormalization>
void Serialization()
{
  DecompositionPolicy decomposition;
  // Load a dataset to train on.
  arma::mat dataset;
  if (!data::Load("GroupLensSmall.csv", dataset))
    FAIL("Cannot load test dataset GroupLensSmall.csv!");

  arma::sp_mat cleanedData;
  CFType<DecompositionPolicy,
      NormalizationType>::CleanData(dataset, cleanedData);

  CFType<DecompositionPolicy,
      NormalizationType> c(cleanedData, decomposition, 5, 5, 30);

  arma::sp_mat randomData;
  randomData.sprandu(100, 100, 0.3);

  CFType<DecompositionPolicy,
      NormalizationType> cXml(randomData, decomposition, 5, 5, 30);
  CFType<DecompositionPolicy,
      NormalizationType> cBinary;
  CFType<DecompositionPolicy,
      NormalizationType> cText(cleanedData, decomposition, 5, 5, 30);

  SerializeObjectAll(c, cXml, cText, cBinary);

  // Check the internals.
  REQUIRE(c.NumUsersForSimilarity() == cXml.NumUsersForSimilarity());
  REQUIRE(c.NumUsersForSimilarity() == cBinary.NumUsersForSimilarity());
  REQUIRE(c.NumUsersForSimilarity() == cText.NumUsersForSimilarity());

  REQUIRE(c.Rank() == cXml.Rank());
  REQUIRE(c.Rank() == cBinary.Rank());
  REQUIRE(c.Rank() == cText.Rank());

  CheckMatrices(c.Decomposition().W(), cXml.Decomposition().W(),
      cBinary.Decomposition().W(), cText.Decomposition().W());
  CheckMatrices(c.Decomposition().H(), cXml.Decomposition().H(),
      cBinary.Decomposition().H(), cText.Decomposition().H());

  REQUIRE(c.CleanedData().n_rows == cXml.CleanedData().n_rows);
  REQUIRE(c.CleanedData().n_rows == cBinary.CleanedData().n_rows);
  REQUIRE(c.CleanedData().n_rows == cText.CleanedData().n_rows);

  REQUIRE(c.CleanedData().n_cols == cXml.CleanedData().n_cols);
  REQUIRE(c.CleanedData().n_cols == cBinary.CleanedData().n_cols);
  REQUIRE(c.CleanedData().n_cols == cText.CleanedData().n_cols);

  REQUIRE(c.CleanedData().n_nonzero == cXml.CleanedData().n_nonzero);
  REQUIRE(c.CleanedData().n_nonzero == cBinary.CleanedData().n_nonzero);
  REQUIRE(c.CleanedData().n_nonzero == cText.CleanedData().n_nonzero);

  c.CleanedData().sync();

  for (size_t i = 0; i <= c.CleanedData().n_cols; ++i)
  {
    REQUIRE(c.CleanedData().col_ptrs[i] == cXml.CleanedData().col_ptrs[i]);
    REQUIRE(c.CleanedData().col_ptrs[i] == cBinary.CleanedData().col_ptrs[i]);
    REQUIRE(c.CleanedData().col_ptrs[i] == cText.CleanedData().col_ptrs[i]);
  }

  for (size_t i = 0; i <= c.CleanedData().n_nonzero; ++i)
  {
    REQUIRE(c.CleanedData().row_indices[i] ==
        cXml.CleanedData().row_indices[i]);
    REQUIRE(c.CleanedData().row_indices[i] ==
        cBinary.CleanedData().row_indices[i]);
    REQUIRE(c.CleanedData().row_indices[i] ==
        cText.CleanedData().row_indices[i]);

    REQUIRE(c.CleanedData().values[i] ==
      Approx(cXml.CleanedData().values[i]).epsilon(1e-7));
    REQUIRE(c.CleanedData().values[i] ==
      Approx(cBinary.CleanedData().values[i]).epsilon(1e-7));
    REQUIRE(c.CleanedData().values[i] ==
      Approx(cText.CleanedData().values[i]).epsilon(1e-7));
  }
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set for randomized SVD.
 */
TEST_CASE("CFGetRecommendationsAllUsersRandSVDTest", "[CFTest]")
{
  GetRecommendationsAllUsers<RandomizedSVDPolicy>();
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set for regularized SVD.
 */
TEST_CASE("CFGetRecommendationsAllUsersRegSVDTest", "[CFTest]")
{
  GetRecommendationsAllUsers<RegSVDPolicy>();
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set for Batch SVD.
 */

TEST_CASE("CFGetRecommendationsAllUsersBatchSVDTest", "[CFTest]")
{
  GetRecommendationsAllUsers<BatchSVDPolicy>();
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set for NMF.
 */
TEST_CASE("CFGetRecommendationsAllUsersNMFTest", "[CFTest]")
{
  GetRecommendationsAllUsers<NMFPolicy>();
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set for SVD Complete Incremental method.
 */
TEST_CASE("CFGetRecommendationsAllUsersSVDCompleteTest", "[CFTest]")
{
  GetRecommendationsAllUsers<SVDCompletePolicy>();
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set for SVD Incomplete Incremental method.
 */
TEST_CASE("CFGetRecommendationsAllUsersSVDIncompleteTest", "[CFTest]")
{
  GetRecommendationsAllUsers<SVDIncompletePolicy>();
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set for Bias SVD method.
 */
TEST_CASE("CFGetRecommendationsAllUsersBiasSVDTest", "[CFTest]")
{
  GetRecommendationsAllUsers<BiasSVDPolicy>();
}

/**
 * Make sure that correct number of recommendations are generated when query
 * set for SVDPlusPlus method.
 */
TEST_CASE("CFGetRecommendationsAllUsersSVDPPTest", "[CFTest]")
{
  GetRecommendationsAllUsers<SVDPlusPlusPolicy>();
}

/**
 * Make sure that the recommendations are generated for queried users only
 * for randomized SVD.
 */
TEST_CASE("CFGetRecommendationsQueriedUserRandSVDTest", "[CFTest]")
{
  GetRecommendationsQueriedUser<RandomizedSVDPolicy>();
}

/**
 * Make sure that the recommendations are generated for queried users only
 * for regularized SVD.
 */
TEST_CASE("CFGetRecommendationsQueriedUserRegSVDTest", "[CFTest]")
{
  GetRecommendationsQueriedUser<RegSVDPolicy>();
}

/**
 * Make sure that the recommendations are generated for queried users only
 * for batch SVD.
 */
TEST_CASE("CFGetRecommendationsQueriedUserBatchSVDTest", "[CFTest]")
{
  GetRecommendationsQueriedUser<BatchSVDPolicy>();
}

/**
 * Make sure that the recommendations are generated for queried users only
 * for NMF.
 */
TEST_CASE("CFGetRecommendationsQueriedUserNMFTest", "[CFTest]")
{
  GetRecommendationsQueriedUser<NMFPolicy>();
}

/**
 * Make sure that the recommendations are generated for queried users only
 * for SVD Complete Incremental method.
 */
TEST_CASE("CFGetRecommendationsQueriedUserSVDCompleteTest", "[CFTest]")
{
  GetRecommendationsQueriedUser<SVDCompletePolicy>();
}

/**
 * Make sure that the recommendations are generated for queried users only
 * for SVD Incomplete Incremental method.
 */
TEST_CASE("CFGetRecommendationsQueriedUserSVDIncompleteTest", "[CFTest]")
{
  GetRecommendationsQueriedUser<SVDIncompletePolicy>();
}

/**
 * Make sure that the recommendations are generated for queried users only
 * for Bias SVD method.
 */
TEST_CASE("CFGetRecommendationsQueriedUserBiasSVDTest", "[CFTest]")
{
  GetRecommendationsQueriedUser<BiasSVDPolicy>();
}

/**
 * Make sure that the recommendations are generated for queried users only
 * for SVDPlusPlus method.
 */
TEST_CASE("CFGetRecommendationsQueriedUserSVDPPTest", "[CFTest]")
{
  GetRecommendationsQueriedUser<SVDPlusPlusPolicy>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for randomized SVD.
 */
TEST_CASE("RecommendationAccuracyRandSVDTest", "[CFTest]")
{
  RecommendationAccuracy<RandomizedSVDPolicy>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for regularized SVD.
 */
TEST_CASE("RecommendationAccuracyRegSVDTest", "[CFTest]")
{
  RecommendationAccuracy<RegSVDPolicy>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for batch SVD.
 */
TEST_CASE("RecommendationAccuracyBatchSVDTest", "[CFTest]")
{
  RecommendationAccuracy<BatchSVDPolicy>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for NMF.
 */
TEST_CASE("RecommendationAccuracyNMFTest", "[CFTest]")
{
  RecommendationAccuracy<NMFPolicy>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for SVD Complete Incremental method.
 */
TEST_CASE("RecommendationAccuracySVDCompleteTest", "[CFTest]")
{
  RecommendationAccuracy<SVDCompletePolicy>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for SVD Incomplete Incremental method.
 */
TEST_CASE("RecommendationAccuracySVDIncompleteTest", "[CFTest]")
{
  RecommendationAccuracy<SVDIncompletePolicy>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for Bias SVD method.
 */
TEST_CASE("RecommendationAccuracyBiasSVDTest", "[CFTest]")
{
  // This algorithm seems to be far less effective than others.
  // We therefore allow failures on 44% of the runs.
  RecommendationAccuracy<BiasSVDPolicy>(22);
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for SVDPlusPlus method.
 */
// This test is commented out because it fails and we haven't solved it yet.
// Please refer to issue #1501 for more info about this test.
// TEST_CASE("RecommendationAccuracySVDPPTest", "[CFTest]")
// {
//   RecommendationAccuracy<SVDPlusPlusPolicy>();
// }

// Make sure that Predict() is returning reasonable results for randomized SVD.
TEST_CASE("CFPredictRandSVDTest", "[CFTest]")
{
  CFPredict<RandomizedSVDPolicy>();
}

// Make sure that Predict() is returning reasonable results for regularized SVD.
TEST_CASE("CFPredictRegSVDTest", "[CFTest]")
{
  CFPredict<RegSVDPolicy>();
}

// Make sure that Predict() is returning reasonable results for batch SVD.
TEST_CASE("CFPredictBatchSVDTest", "[CFTest]")
{
  CFPredict<BatchSVDPolicy>();
}

// Make sure that Predict() is returning reasonable results for NMF.
TEST_CASE("CFPredictNMFTest", "[CFTest]")
{
  CFPredict<NMFPolicy>();
}

/**
 * Make sure that Predict() is returning reasonable results for SVD Complete
 * Incremental method.
 */
TEST_CASE("CFPredictSVDCompleteTest", "[CFTest]")
{
  CFPredict<SVDCompletePolicy>();
}

/**
 * Make sure that Predict() is returning reasonable results for SVD Incomplete
 * Incremental method.
 */
TEST_CASE("CFPredictSVDIncompleteTest", "[CFTest]")
{
  CFPredict<SVDIncompletePolicy>();
}

/**
 * Make sure that Predict() is returning reasonable results for Bias SVD
 * method.
 */
TEST_CASE("CFPredictBiasSVDTest", "[CFTest]")
{
  CFPredict<BiasSVDPolicy>();
}

/**
 * Make sure that Predict() is returning reasonable results for SVDPlusPlus
 * method.
 */
TEST_CASE("CFPredictSVDPPTest", "[CFTest]")
{
  CFPredict<SVDPlusPlusPolicy>();
}

// Compare batch Predict() and individual Predict() for randomized SVD.
TEST_CASE("CFBatchPredictRandSVDTest", "[CFTest]")
{
  BatchPredict<RandomizedSVDPolicy>();
}

// Compare batch Predict() and individual Predict() for regularized SVD.
TEST_CASE("CFBatchPredictRegSVDTest", "[CFTest]")
{
  BatchPredict<RegSVDPolicy>();
}

// Compare batch Predict() and individual Predict() for batch SVD.
TEST_CASE("CFBatchPredictBatchSVDTest", "[CFTest]")
{
  BatchPredict<BatchSVDPolicy>();
}

// Compare batch Predict() and individual Predict() for NMF.
TEST_CASE("CFBatchPredictNMFTest", "[CFTest]")
{
  BatchPredict<NMFPolicy>();
}

// Compare batch Predict() and individual Predict() for
// SVD Complete Incremental method.
TEST_CASE("CFBatchPredictSVDCompleteTest", "[CFTest]")
{
  BatchPredict<SVDCompletePolicy>();
}

// Compare batch Predict() and individual Predict() for
// SVD Incomplete Incremental method.
TEST_CASE("CFBatchPredictSVDIncompleteTest", "[CFTest]")
{
  BatchPredict<SVDIncompletePolicy>();
}

// Compare batch Predict() and individual Predict() for
// Bias SVD method.
TEST_CASE("CFBatchPredictBiasSVDTest", "[CFTest]")
{
  BatchPredict<BiasSVDPolicy>();
}

// Compare batch Predict() and individual Predict() for
// SVDPlusPlus method.
TEST_CASE("CFBatchPredictSVDPPTest", "[CFTest]")
{
  BatchPredict<SVDPlusPlusPolicy>();
}

/**
 * Make sure we can train an already-trained model and it works okay for
 * randomized SVD.
 */
TEST_CASE("TrainRandSVDTest", "[CFTest]")
{
  RandomizedSVDPolicy decomposition;
  Train(decomposition);
}

/**
 * Make sure we can train an already-trained model and it works okay for
 * regularized SVD.
 */
TEST_CASE("TrainRegSVDTest", "[CFTest]")
{
  RegSVDPolicy decomposition;
  TrainWithCoordinateList(decomposition);
}

/**
 * Make sure we can train an already-trained model and it works okay for
 * batch SVD.
 */
TEST_CASE("TrainBatchSVDTest", "[CFTest]")
{
  BatchSVDPolicy decomposition;
  Train(decomposition);
}

/**
 * Make sure we can train an already-trained model and it works okay for
 * NMF.
 */
TEST_CASE("TrainNMFTest", "[CFTest]")
{
  NMFPolicy decomposition;
  Train(decomposition);
}

/**
 * Make sure we can train an already-trained model and it works okay for
 * SVD Complete Incremental method.
 */
TEST_CASE("TrainSVDCompleteTest", "[CFTest]")
{
  SVDCompletePolicy decomposition;
  Train(decomposition);
}

/**
 * Make sure we can train an already-trained model and it works okay for
 * SVD Incomplete Incremental method.
 */
TEST_CASE("TrainSVDIncompleteTest", "[CFTest]")
{
  SVDIncompletePolicy decomposition;
  Train(decomposition);
}

/**
 * Make sure we can train an already-trained model and it works okay for
 * BiasSVD method.
 */
TEST_CASE("TrainBiasSVDTest", "[CFTest]")
{
  BiasSVDPolicy decomposition;
  TrainWithCoordinateList(decomposition);
}

/**
 * Make sure we can train an already-trained model and it works okay for
 * SVDPlusPlus method.
 */
TEST_CASE("TrainSVDPPTest", "[CFTest]")
{
  SVDPlusPlusPolicy decomposition;
  TrainWithCoordinateList(decomposition);
}

/**
 * Make sure we can train a model after using the empty constructor when
 * using randomized SVD.
 */
TEST_CASE("EmptyConstructorTrainRandSVDTest", "[CFTest]")
{
  EmptyConstructorTrain<RandomizedSVDPolicy>();
}

/**
 * Make sure we can train a model after using the empty constructor when
 * using regularized SVD.
 */
TEST_CASE("EmptyConstructorTrainRegSVDTest", "[CFTest]")
{
  EmptyConstructorTrain<RegSVDPolicy>();
}

/**
 * Make sure we can train a model after using the empty constructor when
 * using batch SVD.
 */
TEST_CASE("EmptyConstructorTrainBatchSVDTest", "[CFTest]")
{
  EmptyConstructorTrain<BatchSVDPolicy>();
}

/**
 * Make sure we can train a model after using the empty constructor when
 * using NMF.
 */
TEST_CASE("EmptyConstructorTrainNMFTest", "[CFTest]")
{
  EmptyConstructorTrain<NMFPolicy>();
}

/**
 * Make sure we can train a model after using the empty constructor when
 * using SVD Complete Incremental method.
 */
TEST_CASE("EmptyConstructorTrainSVDCompleteTest", "[CFTest]")
{
  EmptyConstructorTrain<SVDCompletePolicy>();
}

/**
 * Make sure we can train a model after using the empty constructor when
 * using SVD Incomplete Incremental method.
 */
TEST_CASE("EmptyConstructorTrainSVDIncompleteTest", "[CFTest]")
{
  EmptyConstructorTrain<SVDIncompletePolicy>();
}

/**
 * Ensure we can load and save the CF model using randomized SVD policy.
 */
TEST_CASE("SerializationRandSVDTest", "[CFTest]")
{
  Serialization<RandomizedSVDPolicy>();
}

/**
 * Ensure we can load and save the CF model using batch SVD policy.
 */
TEST_CASE("SerializationBatchSVDTest", "[CFTest]")
{
  Serialization<BatchSVDPolicy>();
}

/**
 * Ensure we can load and save the CF model using NMF policy.
 */
TEST_CASE("SerializationNMFTest", "[CFTest]")
{
  Serialization<NMFPolicy>();
}

/**
 * Ensure we can load and save the CF model using SVD Complete Incremental.
 */
TEST_CASE("SerializationSVDCompleteTest", "[CFTest]")
{
  Serialization<SVDCompletePolicy>();
}

/**
 * Ensure we can load and save the CF model using SVD Incomplete Incremental.
 */
TEST_CASE("SerializationSVDIncompleteTest", "[CFTest]")
{
  Serialization<SVDIncompletePolicy>();
}

/**
 * Make sure that Predict() is returning reasonable results for NMF and
 * OverallMeanNormalization.
 */
TEST_CASE("CFPredictOverallMeanNormalization", "[CFTest]")
{
  CFPredict<NMFPolicy, OverallMeanNormalization>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for NMF and
 * UserMeanNormalization.
 */
TEST_CASE("CFPredictUserMeanNormalization", "[CFTest]")
{
  CFPredict<NMFPolicy, UserMeanNormalization>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for NMF and
 * ItemMeanNormalization.
 */
TEST_CASE("CFPredictItemMeanNormalization", "[CFTest]")
{
  CFPredict<NMFPolicy, ItemMeanNormalization>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for NMF and
 * ZScoreNormalization.
 */
TEST_CASE("CFPredictZScoreNormalization", "[CFTest]")
{
  CFPredict<NMFPolicy, ZScoreNormalization>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for NMF and
 * CombinedNormalization<OverallMeanNormalization, UserMeanNormalization,
 * ItemMeanNormalization>.
 */
TEST_CASE("CFPredictCombinedNormalization", "[CFTest]")
{
  CFPredict<NMFPolicy,
            CombinedNormalization<
                OverallMeanNormalization,
                UserMeanNormalization,
                ItemMeanNormalization>>(2.0);
}

/**
 * Make sure that Predict() works with NoNormalization.
 */
TEST_CASE("CFPredictNoNormalization", "[CFTest]")
{
  CFPredict<RegSVDPolicy, NoNormalization>(2.0);
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for OverallMeanNormalization.
 */
TEST_CASE("RecommendationAccuracyOverallMeanNormalizationTest", "[CFTest]")
{
  RecommendationAccuracy<NMFPolicy, OverallMeanNormalization>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for UserMeanNormalization.
 */
TEST_CASE("RecommendationAccuracyUserMeanNormalizationTest", "[CFTest]")
{
  RecommendationAccuracy<NMFPolicy, UserMeanNormalization>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for ItemMeanNormalization.
 */
TEST_CASE("RecommendationAccuracyItemMeanNormalizationTest", "[CFTest]")
{
  RecommendationAccuracy<NMFPolicy, ItemMeanNormalization>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for ZScoreNormalization.
 */
TEST_CASE("RecommendationAccuracyZScoreNormalizationTest", "[CFTest]")
{
  RecommendationAccuracy<NMFPolicy, ZScoreNormalization>();
}

/**
 * Make sure recommendations that are generated are reasonably accurate
 * for CombinedNormalization.
 */
TEST_CASE("RecommendationAccuracyCombinedNormalizationTest", "[CFTest]")
{
  RecommendationAccuracy<NMFPolicy,
                         CombinedNormalization<
                           OverallMeanNormalization,
                           UserMeanNormalization,
                           ItemMeanNormalization>>();
}

/**
 * Ensure we can load and save the CF model using OverallMeanNormalization.
 */
TEST_CASE("SerializationOverallMeanNormalizationTest", "[CFTest]")
{
  Serialization<NMFPolicy, OverallMeanNormalization>();
}

/**
 * Ensure we can load and save the CF model using UserMeanNormalization.
 */
TEST_CASE("SerializationUserMeanNormalizationTest", "[CFTest]")
{
  Serialization<NMFPolicy, UserMeanNormalization>();
}

/**
 * Ensure we can load and save the CF model using ItemMeanNormalization.
 */
TEST_CASE("SerializationItemMeanNormalizationTest", "[CFTest]")
{
  Serialization<NMFPolicy, ItemMeanNormalization>();
}

/**
 * Ensure we can load and save the CF model using ZScoreMeanNormalization.
 */
TEST_CASE("SerializationZScoreNormalizationTest", "[CFTest]")
{
  Serialization<NMFPolicy, ZScoreNormalization>();
}

/**
 * Ensure we can load and save the CF model using CombinedNormalization.
 */
TEST_CASE("SerializationCombinedNormalizationTest", "[CFTest]")
{
  Serialization<NMFPolicy,
                CombinedNormalization<
                    OverallMeanNormalization,
                    UserMeanNormalization,
                    ItemMeanNormalization>>();
}

/**
 * Make sure that Predict() is returning reasonable results for
 * EuclideanSearch.
 */
TEST_CASE("CFPredictEuclideanSearch", "[CFTest]")
{
  CFPredict<NMFPolicy, OverallMeanNormalization, EuclideanSearch>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for
 * CosineSearch.
 */
TEST_CASE("CFPredictCosineSearch", "[CFTest]")
{
  CFPredict<NMFPolicy, OverallMeanNormalization, CosineSearch>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for
 * PearsonSearch.
 */
TEST_CASE("CFPredictPearsonSearch", "[CFTest]")
{
  CFPredict<NMFPolicy, OverallMeanNormalization, PearsonSearch>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for
 * AverageInterpolation.
 */
TEST_CASE("CFPredictAverageInterpolation", "[CFTest]")
{
  CFPredict<NMFPolicy,
            OverallMeanNormalization,
            EuclideanSearch,
            AverageInterpolation>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for
 * SimilarityInterpolation.
 */
TEST_CASE("CFPredictSimilarityInterpolation", "[CFTest]")
{
  CFPredict<NMFPolicy,
            OverallMeanNormalization,
            EuclideanSearch,
            SimilarityInterpolation>(2.0);
}

/**
 * Make sure that Predict() is returning reasonable results for
 * RegressionInterpolation.
 */
TEST_CASE("CFPredictRegressionInterpolation", "[CFTest]")
{
  // Larger tolerance is sometimes needed.
  CFPredict<RegSVDPolicy,
            OverallMeanNormalization,
            EuclideanSearch,
            RegressionInterpolation>(2.2);
}
