/**
 * @file tests/svdplusplus_test.cpp
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
#include <mlpack/methods/svdplusplus.hpp>

#include "catch.hpp"

using namespace mlpack;

TEST_CASE("SVDPlusPlusEvaluate", "[SVDPlusPlusTest]")
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 100;
  const size_t numRatings = 1000;
  const size_t maxRating = 5;
  const size_t rank = 5;
  const size_t numTrials = 10;

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);
  data.row(2) = floor(data.row(2) * maxRating + 0.5);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Make a random implicit dataset.
  arma::sp_mat implicitData = arma::sprandu(numItems, numUsers, 0.1);

  // Make a SVDPlusPlusFunction with zero regularization.
  SVDPlusPlusFunction<arma::mat> svdPPFunc(data, implicitData, rank, 0);

  for (size_t i = 0; i < numTrials; ++i)
  {
    arma::mat parameters = arma::randu(rank + 1, numUsers + 2 * numItems);

    // Calculate cost by summing up cost of each example.
    double cost = 0;
    for (size_t j = 0; j < data.n_cols; ++j)
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
      arma::sp_mat::const_iterator it = implicitData.begin_col(user);
      arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
      size_t implicitCount = 0;
      for (; it != it_end; ++it)
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
    REQUIRE(cost == Approx(svdPPFunc.Evaluate(parameters)).epsilon(1e-7));
  }
}

TEST_CASE("SVDPlusPlusFunctionRegularizationEvaluate", "[SVDPlusPlusTest]")
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 100;
  const size_t numRatings = 1000;
  const size_t maxRating = 5;
  const size_t rank = 5;
  const size_t numTrials = 10;

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);
  data.row(2) = floor(data.row(2) * maxRating + 0.5);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Make a random implicit dataset.
  arma::sp_mat implicitData = arma::sprandu(numItems, numUsers, 0.1);

  // Make three SVDPlusPlusFunction objects with different amounts of
  // regularization.
  SVDPlusPlusFunction<arma::mat> svdPPFuncNoReg(data, implicitData, rank, 0);
  SVDPlusPlusFunction<arma::mat> svdPPFuncSmallReg(data, implicitData, rank,
      0.5);
  SVDPlusPlusFunction<arma::mat> svdPPFuncBigReg(data, implicitData, rank, 20);

  for (size_t i = 0; i < numTrials; ++i)
  {
    arma::mat parameters = arma::randu(rank + 1, numUsers + 2 * numItems);

    // The norm square of implicit item vectors is cached to avoid repeated
    // calculation.
    arma::vec implicitVecsNormSquare(numItems);
    implicitVecsNormSquare.fill(-1);

    // Calculate the regularization contributions of parameters corresponding to
    // each rating and sum them up.
    double smallRegTerm = 0;
    double bigRegTerm = 0;
    for (size_t j = 0; j < data.n_cols; ++j)
    {
      const size_t user = data(0, j);
      const size_t item = data(1, j) + numUsers;
      const size_t implicitStart = numUsers + numItems;

      // Iterate through each item which the user interacted with.
      arma::sp_mat::const_iterator it = implicitData.begin_col(user);
      arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
      double regularizationError = 0;
      size_t implicitCount = 0;
      for (; it != it_end; ++it)
      {
        if (implicitVecsNormSquare(it.row()) < 0)
        {
          implicitVecsNormSquare(it.row()) = arma::dot(
            parameters.col(implicitStart + it.row()).subvec(0, rank - 1),
            parameters.col(implicitStart + it.row()).subvec(0, rank - 1));
        }
        regularizationError += implicitVecsNormSquare(it.row());
        implicitCount += 1;
      }
      if (implicitCount != 0)
        regularizationError /= implicitCount;

      // Calculate the regularization penalty corresponding to the parameters.
      double userVecNorm = arma::norm(parameters.col(user), 2);
      double itemVecNorm = arma::norm(parameters.col(item), 2);
      regularizationError +=
          userVecNorm * userVecNorm + itemVecNorm * itemVecNorm;

      smallRegTerm += 0.5 * regularizationError;
      bigRegTerm += 20 * regularizationError;
    }

    // Cost with regularization should be close to the sum of cost without
    // regularization and the regularization terms.
    REQUIRE(svdPPFuncNoReg.Evaluate(parameters) + smallRegTerm ==
        Approx(svdPPFuncSmallReg.Evaluate(parameters)).epsilon(1e-7));
    REQUIRE(svdPPFuncNoReg.Evaluate(parameters) + bigRegTerm ==
        Approx(svdPPFuncBigReg.Evaluate(parameters)).epsilon(1e-7));
  }
}

TEST_CASE("SVDPlusPlusFunctionGradient", "[SVDPlusPlusTest]")
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 100;
  const size_t numRatings = 1000;
  const size_t maxRating = 5;
  const size_t rank = 5;

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);
  data.row(2) = floor(data.row(2) * maxRating + 0.5);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Make a random implicit dataset.
  arma::sp_mat implicitData = arma::sprandu(numItems, numUsers, 0.1);

  arma::mat parameters = arma::randu(rank + 1, numUsers + 2 * numItems);

  // Make two SVDPlusPlusFunction objects, one with regularization and one
  // without.
  SVDPlusPlusFunction<arma::mat> svdPPFunc1(data, implicitData, rank, 0);
  SVDPlusPlusFunction<arma::mat> svdPPFunc2(data, implicitData, rank, 0.5);

  // Calculate gradients for both the objects.
  arma::mat gradient1, gradient2;
  svdPPFunc1.Gradient(parameters, gradient1);
  svdPPFunc2.Gradient(parameters, gradient2);

  // Perturbation constant.
  const double epsilon = 0.0001;
  double costPlus1, costMinus1, numGradient1;
  double costPlus2, costMinus2, numGradient2;

  for (size_t i = 0; i < rank; ++i)
  {
    for (size_t j = 0; j < numUsers + numItems; ++j)
    {
      // Perturb parameter with a positive constant and get costs.
      parameters(i, j) += epsilon;
      costPlus1 = svdPPFunc1.Evaluate(parameters);
      costPlus2 = svdPPFunc2.Evaluate(parameters);

      // Perturb parameter with a negative constant and get costs.
      parameters(i, j) -= 2 * epsilon;
      costMinus1 = svdPPFunc1.Evaluate(parameters);
      costMinus2 = svdPPFunc2.Evaluate(parameters);

      // Compute numerical gradients using the costs calculated above.
      numGradient1 = (costPlus1 - costMinus1) / (2 * epsilon);
      numGradient2 = (costPlus2 - costMinus2) / (2 * epsilon);

      // Restore the parameter value.
      parameters(i, j) += epsilon;

      // Compare numerical and backpropagation gradient values.
      if (std::abs(gradient1(i, j)) <= 1e-6)
        REQUIRE(numGradient1 == Approx(0.0).margin(1e-5));
      else
      REQUIRE(numGradient1 == Approx(gradient1(i, j)).epsilon(0.0002));

      if (std::abs(gradient2(i, j)) <= 1e-6)
        REQUIRE(numGradient2 == Approx(0.0).margin(1e-5));
      else
        REQUIRE(numGradient2 == Approx(gradient2(i, j)).epsilon(0.0002));
    }
  }
}

TEST_CASE("SVDplusPlusOutputSizeTest", "[SVDPlusPlusTest]")
{
  // Load small GroupLens dataset.
  arma::mat data;
  if (!data::Load("GroupLensSmall.csv", data))
    FAIL("Cannot load dataset GroupLensSmall.csv");

  // Define useful constants.
  const size_t numUsers = max(data.row(0)) + 1;
  const size_t numItems = max(data.row(1)) + 1;
  const size_t rank = 10;
  const size_t iterations = 10;

  // Resulting user/item matrices/bias, and item implicit matrix.
  arma::mat userLatent, itemLatent;
  arma::vec userBias, itemBias;
  arma::mat itemImplicit;

  // Apply SVD++.
  SVDPlusPlus<> svdPP(iterations);
  svdPP.Apply(data, rank, itemLatent, userLatent, itemBias, userBias,
      itemImplicit);

  // Check the size of outputs.
  REQUIRE(itemLatent.n_rows == numItems);
  REQUIRE(itemLatent.n_cols == rank);
  REQUIRE(userLatent.n_rows == rank);
  REQUIRE(userLatent.n_cols == numUsers);
  REQUIRE(itemBias.n_elem == numItems);
  REQUIRE(userBias.n_elem == numUsers);
  REQUIRE(itemImplicit.n_rows == rank);
  REQUIRE(itemImplicit.n_cols == numItems);
}

TEST_CASE("SVDPlusPlusCleanDataTest", "[SVDPlusPlusTest]")
{
  // Load small GroupLens dataset.
  arma::mat data;
  if (!data::Load("GroupLensSmall.csv", data))
    FAIL("Cannot load dataset GroupLensSmall.csv");

  // Define useful constants.
  const size_t numUsers = max(data.row(0)) + 1;
  const size_t numItems = max(data.row(1)) + 1;

  // Make an implicit dataset with the explicit rating dataset.
  arma::mat implicitData = data.submat(0, 0, 1, data.n_cols - 1);

  // We also want to test whether CleanData() can give matrix
  // of right size when maximum user/item is not in implicitData.
  for (size_t i = 0; i < implicitData.n_cols;)
  {
    if (implicitData(0, i) == numUsers - 1 ||
        implicitData(1, i) == numItems - 1)
    {
      implicitData.shed_col(i);
    }
    else
    {
      ++i;
    }
  }

  // Converts implicit data from coordinate list to sparse matrix.
  arma::sp_mat cleanedData;
  SVDPlusPlus<>::CleanData(implicitData, cleanedData, data);

  // Make sure cleanedData has correct size.
  REQUIRE(cleanedData.n_rows == numItems);
  REQUIRE(cleanedData.n_cols == numUsers);

  // Make sure cleanedData has correct number of implicit data.
  REQUIRE(cleanedData.n_nonzero == implicitData.n_cols);

  // Make sure all implicitData are in cleanedData.
  for (size_t i = 0; i < implicitData.n_cols; ++i)
  {
    double value = cleanedData(implicitData(1, i), implicitData(0, i));
    REQUIRE(std::fabs(value) > 0);
  }
}

TEST_CASE("SVDPlusPlusFunctionOptimize", "[SVDPlusPlusTest]")
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 100;
  const size_t numRatings = 1000;
  const size_t iterations = 30;
  const size_t rank = 5;
  const double alpha = 0.01;
  const double lambda = 0;

  // Initiate random parameters.
  arma::mat parameters = arma::randu(rank + 1, numUsers + 2 * numItems);

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Make a random implicit dataset.
  arma::sp_mat implicitData = arma::sprandu(numItems, numUsers, 0.05);

  // Make rating entries based on the parameters.
  for (size_t i = 0; i < numRatings; ++i)
  {
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;
    const size_t implicitStart = numUsers + numItems;

    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);

    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(rank, arma::fill::zeros);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec += parameters.col(implicitStart + it.row()).subvec(0, rank - 1);
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += parameters.col(user).subvec(0, rank - 1);

    data(2, i) = userBias + itemBias +
        arma::dot(userVec, parameters.col(item).subvec(0, rank - 1));
  }

  // Make the SVD++ function and the optimizer.
  SVDPlusPlusFunction<arma::mat> svdPPFunc(data, implicitData, rank, lambda);
  ens::StandardSGD optimizer(alpha, iterations * numRatings);

  // Obtain optimized parameters after training.
  arma::mat optParameters = arma::randu(rank + 1, numUsers + 2 * numItems);
  optimizer.Optimize(svdPPFunc, optParameters);

  // Get predicted ratings from optimized parameters.
  arma::mat predictedData(1, numRatings);
  for (size_t i = 0; i < numRatings; ++i)
  {
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;
    const size_t implicitStart = numUsers + numItems;

    const double userBias = optParameters(rank, user);
    const double itemBias = optParameters(rank, item);

    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(rank, arma::fill::zeros);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec +=
          optParameters.col(implicitStart + it.row()).subvec(0, rank - 1);
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += optParameters.col(user).subvec(0, rank - 1);

    predictedData(0, i) = userBias + itemBias +
        arma::dot(userVec, optParameters.col(item).subvec(0, rank - 1));
  }

  // Calculate relative error.
  const double relativeError = arma::norm(data.row(2) - predictedData, "frob") /
                               arma::norm(data, "frob");

  // Relative error should be small.
  REQUIRE(relativeError == Approx(0.0).margin(1e-2));
}

// The test is only compiled if the user has specified OpenMP to be
// used.
#ifdef MLPACK_USE_OPENMP

// Test SVDPlusPlus with parallel SGD.
TEST_CASE("SVDPlusPlusFunctionParallelOptimize", "[SVDPlusPlusTest]")
{
  // Define useful constants.
  const size_t numUsers = 100;
  const size_t numItems = 100;
  const size_t numRatings = 1000;
  const size_t iterations = 30;
  const size_t rank = 5;
  const double alpha = 0.01;
  const double lambda = 0;

  // Initiate random parameters.
  arma::mat parameters = arma::randu(rank + 1, numUsers + 2 * numItems);

  // Make a random rating dataset.
  arma::mat data = arma::randu(3, numRatings);
  data.row(0) = floor(data.row(0) * numUsers);
  data.row(1) = floor(data.row(1) * numItems);

  // Manually set last row to maximum user and maximum item.
  data(0, numRatings - 1) = numUsers - 1;
  data(1, numRatings - 1) = numItems - 1;

  // Make a random implicit dataset.
  arma::sp_mat implicitData = arma::sprandu(numItems, numUsers, 0.05);

  // Make rating entries based on the parameters.
  for (size_t i = 0; i < numRatings; ++i)
  {
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;
    const size_t implicitStart = numUsers + numItems;

    const double userBias = parameters(rank, user);
    const double itemBias = parameters(rank, item);

    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(rank, arma::fill::zeros);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec += parameters.col(implicitStart + it.row()).subvec(0, rank - 1);
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += parameters.col(user).subvec(0, rank - 1);

    data(2, i) = userBias + itemBias +
        arma::dot(userVec, parameters.col(item).subvec(0, rank - 1));
  }

  // Make the SVD++ function and the optimizer.
  SVDPlusPlusFunction<arma::mat> svdPPFunc(data, implicitData, rank, lambda);

  ens::ConstantStep decayPolicy(alpha);

  // Iterate till convergence.
  // The threadShareSize is chosen such that each function gets optimized.
  ens::ParallelSGD<ens::ConstantStep> optimizer(iterations,
      std::ceil((float) svdPPFunc.NumFunctions() / omp_get_max_threads()), 1e-5,
      true, decayPolicy);

  // Obtain optimized parameters after training.
  arma::mat optParameters = arma::randu(rank + 1, numUsers + 2 * numItems);
  optimizer.Optimize(svdPPFunc, optParameters);

  // Get predicted ratings from optimized parameters.
  arma::mat predictedData(1, numRatings);
  for (size_t i = 0; i < numRatings; ++i)
  {
    const size_t user = data(0, i);
    const size_t item = data(1, i) + numUsers;
    const size_t implicitStart = numUsers + numItems;

    const double userBias = optParameters(rank, user);
    const double itemBias = optParameters(rank, item);

    // Iterate through each item which the user interacted with to calculate
    // user vector.
    arma::vec userVec(rank, arma::fill::zeros);
    arma::sp_mat::const_iterator it = implicitData.begin_col(user);
    arma::sp_mat::const_iterator it_end = implicitData.end_col(user);
    size_t implicitCount = 0;
    for (; it != it_end; ++it)
    {
      userVec +=
          optParameters.col(implicitStart + it.row()).subvec(0, rank - 1);
      implicitCount += 1;
    }
    if (implicitCount != 0)
      userVec /= std::sqrt(implicitCount);
    userVec += optParameters.col(user).subvec(0, rank - 1);

    predictedData(0, i) = userBias + itemBias +
        arma::dot(userVec, optParameters.col(item).subvec(0, rank - 1));
  }

  // Calculate relative error.
  const double relativeError = arma::norm(data.row(2) - predictedData, "frob") /
                               arma::norm(data, "frob");

  // Relative error should be small.
  REQUIRE(relativeError == Approx(0.0).margin(1e-2));
}

#endif
