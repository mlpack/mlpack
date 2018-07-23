/**
 * @file svdplusplus_test.cpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * Test SVD++.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/svdplusplus/svdplusplus.hpp>
#include <mlpack/core/optimizers/parallel_sgd/parallel_sgd.hpp>
#include <mlpack/core/optimizers/parallel_sgd/decay_policies/constant_step.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::svd;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(SVDPlusPlusTest);

BOOST_AUTO_TEST_CASE(SVDPlusPlusRandomEvaluate)
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 100;
  const size_t numRatings = 1000;
  const size_t numImplicitData = 500;
  const size_t maxRating = 5;
  const size_t rank = 10;
  const size_t numTrials = 50;

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);
  data.row(2) = floor(data.row(2) * maxRating + 0.5);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Make a random implicit dataset.
  arma::mat implicitData = arma::randu(2, numImplicitData);
  implicitData.row(0) = floor(implicitData.row(0) * numUsers);
  implicitData.row(1) = floor(implicitData.row(1) * numItems);

  // Converts implicit data from coordinate list to sparse matrix.
  arma::sp_mat cleanedData;
  SVDPlusPlus<>::CleanData(implicitData, cleanedData, data);

  // Make a SVDPlusPlusFunction with zero regularization.
  SVDPlusPlusFunction<arma::mat> svdPPFunc(data, cleanedData, rank, 0);

  for (size_t i = 0; i < numTrials; i++)
  {
    arma::mat parameters = arma::randu(rank + 1, numUsers + 2 * numItems);

    // Calculate cost by summing up cost of each example.
    double cost = 0;
    for (size_t j = 0; j < numRatings; j++)
    {
      const size_t user = data(0, j);
      const size_t item = data(1, j) + numUsers;
      const size_t implicitStart = numUsers + numItems;

      // Calculate the squared error in the prediction.
      const double rating = data(2, j);
      const double userBias = parameters(rank, user);
      const double itemBias = parameters(rank, item);

      // Iterate through each item which the user interacted with to calculate
      // user vector.
      arma::vec userVec(rank, arma::fill::zeros);
      arma::sp_mat::const_iterator it = cleanedData.begin_col(user);
      arma::sp_mat::const_iterator it_end = cleanedData.end_col(user);
      size_t implicitCount = 0;
      for (; it != it_end; it++)
      {
        userVec += parameters.col(implicitStart + it.row()).subvec(0, rank - 1);
        implicitCount += 1;
      }
      if (implicitCount != 0)
        userVec /= std::sqrt(implicitCount);
      userVec += parameters.col(user).subvec(0, rank - 1);

      double ratingError = rating - userBias - itemBias -
          arma::dot(userVec, parameters.col(item).subvec(0, rank - 1));
      double ratingErrorSquared = ratingError * ratingError;

      cost += ratingErrorSquared;
    }

    // Compare calculated cost and value obtained using Evaluate().
    BOOST_REQUIRE_CLOSE(cost, svdPPFunc.Evaluate(parameters), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SVDplusPlusOutputSizeTest)
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 50;
  const size_t numRatings = 500;
  const size_t maxRating = 5;
  const size_t rank = 5;
  const size_t iterations = 10;

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);
  data.row(2) = floor(data.row(2) * maxRating + 0.5);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Resulting user/item matrices/bias, and item implicit matrix.
  arma::mat userLatent, itemLatent;
  arma::vec userBias, itemBias;
  arma::mat itemImplicit;

  // Apply SVD++.
  SVDPlusPlus<> svdPP(iterations);
  svdPP.Apply(data, rank, itemLatent, userLatent, itemBias, userBias, itemImplicit);

  // Check the size of outputs.
  BOOST_REQUIRE_EQUAL(itemLatent.n_rows, numItems);
  BOOST_REQUIRE_EQUAL(itemLatent.n_cols, rank);
  BOOST_REQUIRE_EQUAL(userLatent.n_rows, rank);
  BOOST_REQUIRE_EQUAL(userLatent.n_cols, numUsers);
  BOOST_REQUIRE_EQUAL(itemBias.n_elem, numItems);
  BOOST_REQUIRE_EQUAL(userBias.n_elem, numUsers);
  BOOST_REQUIRE_EQUAL(itemImplicit.n_rows, rank);
  BOOST_REQUIRE_EQUAL(itemImplicit.n_cols, numItems);
}

BOOST_AUTO_TEST_CASE(SVDPlusPlusCleanDataTest)
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 100;
  const size_t numImplicitData = 2000;
  const size_t numRatings = 1000;
  const size_t maxRating = 5;

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);
  data.row(2) = floor(data.row(2) * maxRating + 0.5);

  // Manually set last column to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Make a random implicit dataset.
  arma::mat implicitData = arma::randu(2, numImplicitData);
  implicitData.row(0) = floor(implicitData.row(0) * numUsers);
  implicitData.row(1) = floor(implicitData.row(1) * numItems);

  // We also want to test whether CleanData() can give matrix
  // of right size when maximum user/item is not in implicitData.
  for (size_t i = 0; i < numImplicitData; i++)
  {
    if (implicitData(0, i) == numUsers - 1)
      implicitData(0, i) = 0;
    if (implicitData(1, i) == numItems - 1)
      implicitData(1, i) = 0;
  }

  // Converts implicit data from coordinate list to sparse matrix.
  arma::sp_mat cleanedData;
  SVDPlusPlus<>::CleanData(implicitData, cleanedData, data);

  // Make sure cleanedData has correct size.
  BOOST_REQUIRE_EQUAL(cleanedData.n_rows, numItems);
  BOOST_REQUIRE_EQUAL(cleanedData.n_cols, numUsers);

  // Make sure cleanedData has correct number of implicit data.
  BOOST_REQUIRE_EQUAL(cleanedData.n_nonzero, numImplicitData);

  // Make sure all implicitData are in cleanedData.
  for (size_t i = 0; i < numImplicitData; i++)
  {
    double value = cleanedData(implicitData(1, i), implicitData(0, i));
    BOOST_REQUIRE_GT(std::fabs(value), 1e-5);
  }
}

BOOST_AUTO_TEST_SUITE_END();
