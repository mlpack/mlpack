/**
 * @file regularized_svd_test.cpp
 * @author Siddharth Agrawal
 *
 * Test the RegularizedSVDFunction class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/regularized_svd/regularized_svd.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::svd;

BOOST_AUTO_TEST_SUITE(RegularizedSVDTest);

BOOST_AUTO_TEST_CASE(RegularizedSVDFunctionRandomEvaluate)
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

  // Make a RegularizedSVDFunction with zero regularization.
  RegularizedSVDFunction rSVDFunc(data, rank, 0);

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
      const double ratingError = rating - arma::dot(parameters.col(user),
                                                    parameters.col(item));
      const double ratingErrorSquared = ratingError * ratingError;

      cost += ratingErrorSquared;
    }

    // Compare calculated cost and value obtained using Evaluate().
    BOOST_REQUIRE_CLOSE(cost, rSVDFunc.Evaluate(parameters), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(RegularizedSVDFunctionRegularizationEvaluate)
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

  // Make three RegularizedSVDFunction objects with different amounts of
  // regularization.
  RegularizedSVDFunction rSVDFuncNoReg(data, rank, 0);
  RegularizedSVDFunction rSVDFuncSmallReg(data, rank, 0.5);
  RegularizedSVDFunction rSVDFuncBigReg(data, rank, 20);

  for (size_t i = 0; i < numTrials; i++)
  {
    arma::mat parameters = arma::randu(rank, numUsers + numItems);

    // Calculate the regularization contributions of parameters corresponding to
    // each rating and sum them up.
    double smallRegTerm = 0;
    double bigRegTerm = 0;
    for (size_t j = 0; j < numRatings; j++)
    {
      const size_t user = data(0, j);
      const size_t item = data(1, j) + numUsers;

      const double userVecNorm = arma::norm(parameters.col(user), 2);
      const double itemVecNorm = arma::norm(parameters.col(item), 2);
      smallRegTerm += 0.5 * (userVecNorm * userVecNorm +
                             itemVecNorm * itemVecNorm);
      bigRegTerm += 20 * (userVecNorm * userVecNorm +
                          itemVecNorm * itemVecNorm);
    }

    // Cost with regularization should be close to the sum of cost without
    // regularization and the regularization terms.
    BOOST_REQUIRE_CLOSE(rSVDFuncNoReg.Evaluate(parameters) + smallRegTerm,
        rSVDFuncSmallReg.Evaluate(parameters), 1e-5);
    BOOST_REQUIRE_CLOSE(rSVDFuncNoReg.Evaluate(parameters) + bigRegTerm,
        rSVDFuncBigReg.Evaluate(parameters), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(RegularizedSVDFunctionGradient)
{
  // Define useful constants.
  const size_t numUsers = 50;
  const size_t numItems = 50;
  const size_t numRatings = 100;
  const size_t maxRating = 5;
  const size_t rank = 10;

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);
  data.row(2) = floor(data.row(2) * maxRating + 0.5);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  arma::mat parameters = arma::randu(rank, numUsers + numItems);

  // Make two RegularizedSVDFunction objects, one with regularization and one
  // without.
  RegularizedSVDFunction rSVDFunc1(data, rank, 0);
  RegularizedSVDFunction rSVDFunc2(data, rank, 0.5);

  // Calculate gradients for both the objects.
  arma::mat gradient1, gradient2;
  rSVDFunc1.Gradient(parameters, gradient1);
  rSVDFunc2.Gradient(parameters, gradient2);

  // Perturbation constant.
  const double epsilon = 0.0001;
  double costPlus1, costMinus1, numGradient1;
  double costPlus2, costMinus2, numGradient2;

  for (size_t i = 0; i < rank; i++)
  {
    for (size_t j = 0; j < numUsers + numItems; j++)
    {
      // Perturb parameter with a positive constant and get costs.
      parameters(i, j) += epsilon;
      costPlus1 = rSVDFunc1.Evaluate(parameters);
      costPlus2 = rSVDFunc2.Evaluate(parameters);

      // Perturb parameter with a negative constant and get costs.
      parameters(i, j) -= 2 * epsilon;
      costMinus1 = rSVDFunc1.Evaluate(parameters);
      costMinus2 = rSVDFunc2.Evaluate(parameters);

      // Compute numerical gradients using the costs calculated above.
      numGradient1 = (costPlus1 - costMinus1) / (2 * epsilon);
      numGradient2 = (costPlus2 - costMinus2) / (2 * epsilon);

      // Restore the parameter value.
      parameters(i, j) += epsilon;

      // Compare numerical and backpropagation gradient values.
      if (std::abs(gradient1(i, j)) <= 1e-6)
        BOOST_REQUIRE_SMALL(numGradient1, 1e-5);
      else
        BOOST_REQUIRE_CLOSE(numGradient1, gradient1(i, j), 1e-2);

      if (std::abs(gradient2(i, j)) <= 1e-6)
        BOOST_REQUIRE_SMALL(numGradient2, 1e-5);
      else
        BOOST_REQUIRE_CLOSE(numGradient2, gradient2(i, j), 1e-2);
    }
  }
}

BOOST_AUTO_TEST_CASE(RegularizedSVDFunctionOptimize)
{
  // Define useful constants.
  const size_t numUsers = 50;
  const size_t numItems = 50;
  const size_t numRatings = 100;
  const size_t iterations = 30;
  const size_t rank = 10;
  const double alpha = 0.01;
  const double lambda = 0.01;

  // Initiate random parameters.
  arma::mat parameters = arma::randu(rank, numUsers + numItems);

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Make rating entries based on the parameters.
  for (size_t i = 0; i < numRatings; i++)
  {
    data(2, i) = arma::dot(parameters.col(data(0, i)),
                           parameters.col(numUsers + data(1, i)));
  }

  // Make the Reg SVD function and the optimizer.
  RegularizedSVDFunction rSVDFunc(data, rank, lambda);
  mlpack::optimization::SGD<RegularizedSVDFunction> optimizer(rSVDFunc,
      alpha, iterations * numRatings);

  // Obtain optimized parameters after training.
  arma::mat optParameters = arma::randu(rank, numUsers + numItems);
  optimizer.Optimize(optParameters);

  // Get predicted ratings from optimized parameters.
  arma::mat predictedData(1, numRatings);
  for (size_t i = 0; i < numRatings; i++)
  {
    predictedData(0, i) = arma::dot(optParameters.col(data(0, i)),
                                    optParameters.col(numUsers + data(1, i)));
  }

  // Calculate relative error.
  const double relativeError = arma::norm(data.row(2) - predictedData, "frob") /
                               arma::norm(data, "frob");

  // Relative error should be small.
  BOOST_REQUIRE_SMALL(relativeError, 1e-2);
}

BOOST_AUTO_TEST_SUITE_END();
