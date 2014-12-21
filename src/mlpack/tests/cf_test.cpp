/**
 * @file cf_test.cpp
 * @author Mudit Raj Gupta
 *
 * Test file for CF class.
 *
 * This file is part of MLPACK 1.0.9.
 *
 * MLPACK is free software: you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * MLPACK is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
 * details (LICENSE.txt).
 *
 * You should have received a copy of the GNU General Public License along with
 * MLPACK.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <mlpack/core.hpp>
#include <mlpack/methods/cf/cf.hpp>
#include <iostream>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

BOOST_AUTO_TEST_SUITE(CFTest);

using namespace mlpack;
using namespace mlpack::cf;
using namespace std;

/**
 * Make sure that the constructor works okay.
 */
BOOST_AUTO_TEST_CASE(CFConstructorTest)
{
  // Load GroupLens data.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  // Number of users for similarity (not the default).
  const size_t numUsersForSimilarity = 8;

  CF<> c(dataset, numUsersForSimilarity);

  // Check parameters.
  BOOST_REQUIRE_EQUAL(c.NumUsersForSimilarity(), numUsersForSimilarity);

  // Check data.
  BOOST_REQUIRE_EQUAL(c.Data().n_rows, dataset.n_rows);
  BOOST_REQUIRE_EQUAL(c.Data().n_cols, dataset.n_cols);

  // Check values (this should be superfluous...).
  for (size_t i = 0; i < dataset.n_rows; i++)
    for (size_t j = 0; j < dataset.n_cols; j++)
      BOOST_REQUIRE_EQUAL(c.Data()(i, j), dataset(i, j));
}

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

  // Creat a CF object
  CF<> c(dataset);

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

  CF<> c(dataset);

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

  // Now create the CF object.
  CF<> c(dataset);

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

BOOST_AUTO_TEST_SUITE_END();
