/**
 * @file bias_svd_test.cpp
 * @author Siddharth Agrawal
 * @author Wenhao Huang
 *
 * Test the BiasSVDFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/bias_svd/bias_svd.hpp>
#include <mlpack/core/optimizers/parallel_sgd/parallel_sgd.hpp>
#include <mlpack/core/optimizers/parallel_sgd/decay_policies/constant_step.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::svd;
using namespace mlpack::optimization;

BOOST_AUTO_TEST_SUITE(BiasSVDTest);

BOOST_AUTO_TEST_CASE(BiasSVDFunctionRandomEvaluate)
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 100;
  const size_t numRatings = 1000;
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

  // Make a BiasSVDFunction with zero regularization.
  BiasSVDFunction<arma::mat> rSVDFunc(data, rank, 0);

  for (size_t i = 0; i < numTrials; i++)
  {
    arma::mat parameters = arma::randu(rank, numUsers + numItems);

    // Calculate cost by summing up cost of each example.
    double cost = 0;
    for (size_t j = 0; j < numRatings; j++)
    {
      const size_t user = data(0, j);
      const size_t item = data(1, j) + numUsers;

      const double rating = data(2, j);
      const double userBias = parameters(rank, user);
      const double itemBias = parameters(rank, item);
      const double ratingError = rating - userBias - itemBias -
          arma::dot(parameters.col(user).subvec(0, rank - 1),
                    parameters.col(item).subvec(0, rank - 1));
      const double ratingErrorSquared = ratingError * ratingError;

      cost += ratingErrorSquared;
    }

    // Compare calculated cost and value obtained using Evaluate().
    BOOST_REQUIRE_CLOSE(cost, rSVDFunc.Evaluate(parameters), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(BiasSVDOutputSizeTest)
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

  // Resulting user/item matrices/bias.
  arma::mat userLatent, itemLatent;
  arma::vec userBias, itemBias;

  // Apply Bias SVD.
  BiasSVD<> biasSVD(iterations);
  biasSVD.Apply(data, rank, itemLatent, userLatent, itemBias, userBias);

  // Check the size of outputs.
  BOOST_REQUIRE_EQUAL(itemLatent.n_rows, numItems);
  BOOST_REQUIRE_EQUAL(itemLatent.n_cols, rank);
  BOOST_REQUIRE_EQUAL(userLatent.n_rows, rank);
  BOOST_REQUIRE_EQUAL(userLatent.n_cols, numUsers);
  BOOST_REQUIRE_EQUAL(itemBias.n_elem, numItems);
  BOOST_REQUIRE_EQUAL(userBias.n_elem, numUsers);
}

BOOST_AUTO_TEST_SUITE_END();
