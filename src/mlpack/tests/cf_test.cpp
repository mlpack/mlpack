/**
 * @file cf_test.cpp
 * @author Mudit Raj Gupta
 *
 * Test file for CF class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/cf/cf.hpp>
#include <iostream>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"
#include "serialization.hpp"

BOOST_AUTO_TEST_SUITE(CFTest);

using namespace mlpack;
using namespace mlpack::cf;
using namespace std;

/**
 * Make sure that correct number of recommendations are generated when query
 * set. Default case.
 */
BOOST_AUTO_TEST_CASE(CFGetRecommendationsAllUsersTest)
{
  // Dummy number of recommendations.
  size_t numRecs = 3;
  // GroupLens100k.csv dataset has 943 users.
  size_t numUsers = 943;

  // Matrix to save recommendations into.
  arma::Mat<size_t> recommendations;

  // Load GroupLens data.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Make data into sparse matrix.
  arma::sp_mat cleanedData;
  CF::CleanData(dataset, cleanedData);

  // Create a CF object.
  CF c(cleanedData);

  // Generate recommendations when query set is not specified.
  c.GetRecommendations(numRecs, recommendations);

  // Check if correct number of recommendations are generated.
  BOOST_REQUIRE_EQUAL(recommendations.n_rows, numRecs);

  // Check if recommendations are generated for all users.
  BOOST_REQUIRE_EQUAL(recommendations.n_cols, numUsers);
}

/**
 * Make sure that the recommendations are generated for queried users only.
 */
BOOST_AUTO_TEST_CASE(CFGetRecommendationsQueriedUserTest)
{
  // Number of users that we will search for recommendations for.
  size_t numUsers = 10;

  // Default number of recommendations.
  size_t numRecsDefault = 5;

  // Create dummy query set.
  arma::Col<size_t> users = arma::zeros<arma::Col<size_t> >(numUsers, 1);
  for (size_t i = 0; i < numUsers; i++)
    users(i) = i;

  // Matrix to save recommendations into.
  arma::Mat<size_t> recommendations;

  // Load GroupLens data.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Make data into sparse matrix.
  arma::sp_mat cleanedData;
  CF::CleanData(dataset, cleanedData);

  CF c(cleanedData);

  // Generate recommendations when query set is specified.
  c.GetRecommendations(numRecsDefault, recommendations, users);

  // Check if correct number of recommendations are generated.
  BOOST_REQUIRE_EQUAL(recommendations.n_rows, numRecsDefault);

  // Check if recommendations are generated for the right number of users.
  BOOST_REQUIRE_EQUAL(recommendations.n_cols, numUsers);
}

/**
 * Make sure recommendations that are generated are reasonably accurate.
 */
BOOST_AUTO_TEST_CASE(RecommendationAccuracyTest)
{
  // Load the GroupLens dataset; then, we will remove some values from it.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Save the columns we've removed.
  arma::mat savedCols(3, 300); // Remove 300 5-star ratings.
  size_t currentCol = 0;
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (currentCol == 300)
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

      // If this user doesn't already exist in savedCols, add them.  Otherwise
      // ignore this point.
      if (!found)
      {
        savedCols.col(currentCol) = dataset.col(i);
        dataset.shed_col(i);
        ++currentCol;
      }
    }
  }

  // Make data into sparse matrix.
  arma::sp_mat cleanedData;
  CF::CleanData(dataset, cleanedData);

  // Now create the CF object.
  CF c(cleanedData);

  // Obtain 150 recommendations for the users in savedCols, and make sure the
  // missing item shows up in most of them.  First, create the list of users,
  // which requires casting from doubles...
  arma::Col<size_t> users(300);
  for (size_t i = 0; i < 300; ++i)
    users(i) = (size_t) savedCols(0, i);
  arma::Mat<size_t> recommendations;
  size_t numRecs = 150;
  c.GetRecommendations(numRecs, recommendations, users);

  BOOST_REQUIRE_EQUAL(recommendations.n_rows, numRecs);
  BOOST_REQUIRE_EQUAL(recommendations.n_cols, 300);

  size_t failures = 0;
  for (size_t i = 0; i < 300; ++i)
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
        BOOST_REQUIRE_EQUAL((double) c.CleanedData()(item, user), 0.0);
      }
    }

    if (!found)
      ++failures;
  }

  // Make sure the right item showed up in at least 2/3 of the recommendations.
  // Random chance (that is, if we selected recommendations randomly) for this
  // GroupLens dataset would give somewhere around a 10% success rate (failures
  // would be closer to 270).
  BOOST_REQUIRE_LT(failures, 100);
}

// Make sure that Predict() is returning reasonable results.
BOOST_AUTO_TEST_CASE(CFPredictTest)
{
  // Load the GroupLens dataset; then, we will remove some values from it.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Save the columns we've removed.
  arma::mat savedCols(3, 300); // Remove 300 5-star ratings.
  size_t currentCol = 0;
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (currentCol == 300)
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

      // If this user doesn't already exist in savedCols, add them.  Otherwise
      // ignore this point.
      if (!found)
      {
        savedCols.col(currentCol) = dataset.col(i);
        dataset.shed_col(i);
        ++currentCol;
      }
    }
  }

  // Make data into sparse matrix.
  arma::sp_mat cleanedData;
  CF::CleanData(dataset, cleanedData);

  // Now create the CF object.
  CF c(cleanedData);

  // Now, for each removed rating, make sure the prediction is... reasonably
  // accurate.
  double totalError = 0.0;
  for (size_t i = 0; i < savedCols.n_cols; ++i)
  {
    const double prediction = c.Predict(savedCols(0, i), savedCols(1, i));

    const double error = std::pow(prediction - savedCols(2, i), 2.0);
    totalError += error;
  }

  totalError = std::sqrt(totalError) / savedCols.n_cols;

  // The mean squared error should be less than one.
  BOOST_REQUIRE_LT(totalError, 0.5);
}

// Do the same thing as the previous test, but ensure that the ratings we
// predict with the batch Predict() are the same as the individual Predict()
// calls.
BOOST_AUTO_TEST_CASE(CFBatchPredictTest)
{
  // Load the GroupLens dataset; then, we will remove some values from it.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Save the columns we've removed.
  arma::mat savedCols(3, 300); // Remove 300 5-star ratings.
  size_t currentCol = 0;
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (currentCol == 300)
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

      // If this user doesn't already exist in savedCols, add them.  Otherwise
      // ignore this point.
      if (!found)
      {
        savedCols.col(currentCol) = dataset.col(i);
        dataset.shed_col(i);
        ++currentCol;
      }
    }
  }

  // Make data into sparse matrix.
  arma::sp_mat cleanedData;
  CF::CleanData(dataset, cleanedData);

  // Now create the CF object.
  CF c(cleanedData);

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
    BOOST_REQUIRE_CLOSE(prediction, predictions[i], 1e-8);
  }
}

/**
 * Make sure we can train an already-trained model and it works okay.
 */
BOOST_AUTO_TEST_CASE(TrainTest)
{
  // Generate random data.
  arma::sp_mat randomData;
  randomData.sprandu(100, 100, 0.3);
  CF c(randomData);

  // Now retrain with data we know about.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Save the columns we've removed.
  arma::mat savedCols(3, 300); // Remove 300 5-star ratings.
  size_t currentCol = 0;
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (currentCol == 300)
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

      // If this user doesn't already exist in savedCols, add them.  Otherwise
      // ignore this point.
      if (!found)
      {
        savedCols.col(currentCol) = dataset.col(i);
        dataset.shed_col(i);
        ++currentCol;
      }
    }
  }

  // Make data into sparse matrix.
  arma::sp_mat cleanedData;
  CF::CleanData(dataset, cleanedData);

  // Now retrain.
  c.Train(dataset);

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
    BOOST_REQUIRE_CLOSE(prediction, predictions[i], 1e-8);
  }
}

/**
 * Make sure we can train a model after using the empty constructor.
 */
BOOST_AUTO_TEST_CASE(EmptyConstructorTrainTest)
{
  // Use default constructor.
  CF c;

  // Now retrain with data we know about.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Save the columns we've removed.
  arma::mat savedCols(3, 300); // Remove 300 5-star ratings.
  size_t currentCol = 0;
  for (size_t i = 0; i < dataset.n_cols; ++i)
  {
    if (currentCol == 300)
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

      // If this user doesn't already exist in savedCols, add them.  Otherwise
      // ignore this point.
      if (!found)
      {
        savedCols.col(currentCol) = dataset.col(i);
        dataset.shed_col(i);
        ++currentCol;
      }
    }
  }

  // Make data into sparse matrix.
  arma::sp_mat cleanedData;
  CF::CleanData(dataset, cleanedData);

  // Now retrain.
  c.Train(cleanedData);

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
    BOOST_REQUIRE_CLOSE(prediction, predictions[i], 1e-8);
  }
}

/**
 * Ensure we can load and save the CF model.
 */
BOOST_AUTO_TEST_CASE(SerializationTest)
{
  // Load a dataset to train on.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  arma::sp_mat cleanedData;
  CF::CleanData(dataset, cleanedData);

  CF c(cleanedData);

  arma::sp_mat randomData;
  randomData.sprandu(100, 100, 0.3);

  CF cXml(randomData);
  CF cBinary;
  CF cText(cleanedData, amf::NMFALSFactorizer(), 5, 5);

  SerializeObjectAll(c, cXml, cText, cBinary);

  // Check the internals.
  BOOST_REQUIRE_EQUAL(c.NumUsersForSimilarity(), cXml.NumUsersForSimilarity());
  BOOST_REQUIRE_EQUAL(c.NumUsersForSimilarity(),
      cBinary.NumUsersForSimilarity());
  BOOST_REQUIRE_EQUAL(c.NumUsersForSimilarity(), cText.NumUsersForSimilarity());

  BOOST_REQUIRE_EQUAL(c.Rank(), cXml.Rank());
  BOOST_REQUIRE_EQUAL(c.Rank(), cBinary.Rank());
  BOOST_REQUIRE_EQUAL(c.Rank(), cText.Rank());

  CheckMatrices(c.W(), cXml.W(), cBinary.W(), cText.W());
  CheckMatrices(c.H(), cXml.H(), cBinary.H(), cText.H());

  BOOST_REQUIRE_EQUAL(c.CleanedData().n_rows, cXml.CleanedData().n_rows);
  BOOST_REQUIRE_EQUAL(c.CleanedData().n_rows, cBinary.CleanedData().n_rows);
  BOOST_REQUIRE_EQUAL(c.CleanedData().n_rows, cText.CleanedData().n_rows);

  BOOST_REQUIRE_EQUAL(c.CleanedData().n_cols, cXml.CleanedData().n_cols);
  BOOST_REQUIRE_EQUAL(c.CleanedData().n_cols, cBinary.CleanedData().n_cols);
  BOOST_REQUIRE_EQUAL(c.CleanedData().n_cols, cText.CleanedData().n_cols);

  BOOST_REQUIRE_EQUAL(c.CleanedData().n_nonzero, cXml.CleanedData().n_nonzero);
  BOOST_REQUIRE_EQUAL(c.CleanedData().n_nonzero,
      cBinary.CleanedData().n_nonzero);
  BOOST_REQUIRE_EQUAL(c.CleanedData().n_nonzero, cText.CleanedData().n_nonzero);

  for (size_t i = 0; i <= c.CleanedData().n_cols; ++i)
  {
    BOOST_REQUIRE_EQUAL(c.CleanedData().col_ptrs[i],
        cXml.CleanedData().col_ptrs[i]);
    BOOST_REQUIRE_EQUAL(c.CleanedData().col_ptrs[i],
        cBinary.CleanedData().col_ptrs[i]);
    BOOST_REQUIRE_EQUAL(c.CleanedData().col_ptrs[i],
        cText.CleanedData().col_ptrs[i]);
  }

  for (size_t i = 0; i <= c.CleanedData().n_nonzero; ++i)
  {
    BOOST_REQUIRE_EQUAL(c.CleanedData().row_indices[i],
        cXml.CleanedData().row_indices[i]);
    BOOST_REQUIRE_EQUAL(c.CleanedData().row_indices[i],
        cBinary.CleanedData().row_indices[i]);
    BOOST_REQUIRE_EQUAL(c.CleanedData().row_indices[i],
        cText.CleanedData().row_indices[i]);

    BOOST_REQUIRE_CLOSE(c.CleanedData().values[i], cXml.CleanedData().values[i],
        1e-5);
    BOOST_REQUIRE_CLOSE(c.CleanedData().values[i],
        cBinary.CleanedData().values[i], 1e-5);
    BOOST_REQUIRE_CLOSE(c.CleanedData().values[i],
        cText.CleanedData().values[i], 1e-5);
  }
}


BOOST_AUTO_TEST_SUITE_END();
