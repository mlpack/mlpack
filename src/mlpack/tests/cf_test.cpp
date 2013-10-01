/**
 * @file cf_test.cpp
 * @author Mudit Raj Gupta
 *
 * Test file for CF class.
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

  // Number of recommendations (not the default).
  const size_t numRecs = 15;

  // Number of users for similarity (not the default).
  const size_t numUsersForSimilarity = 8;

  CF c(numRecs, numUsersForSimilarity, dataset);

  // Check parameters.
  BOOST_REQUIRE_EQUAL(c.NumRecs(), numRecs);
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
  CF c(dataset);

  // Set number of recommendations.
  c.NumRecs(numRecs);

  // Generate recommendations when query set is not specified.
  c.GetRecommendations(recommendations);

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

  // Creaate dummy query set.
  arma::Col<size_t> users = arma::zeros<arma::Col<size_t> >(numUsers, 1);
  for (size_t i = 0; i < numUsers; i++)
    users(i) = i + 1;

  // Matrix to save recommendations into.
  arma::Mat<size_t> recommendations;

  // Load GroupLens data.
  arma::mat dataset;
  data::Load("GroupLens100k.csv", dataset);

  CF c(dataset);

  // Generate recommendations when query set is specified.
  c.GetRecommendations(recommendations, users);

  // Check if correct number of recommendations are generated.
  BOOST_REQUIRE_EQUAL(recommendations.n_rows, numRecsDefault);

  // Check if recommendations are generated for the right number of users.
  BOOST_REQUIRE_EQUAL(recommendations.n_cols, numUsers);
}

BOOST_AUTO_TEST_SUITE_END();
