/**
 * @file tests/softmax_regression_test.cpp
 * @author Siddharth Agrawal
 *
 * Test the SoftmaxRegression class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression.hpp>

#include "catch.hpp"

using namespace mlpack;

TEST_CASE("SoftmaxRegressionFunctionEvaluate", "[SoftmaxRegressionTest]")
{
  const size_t points = 1000;
  const size_t trials = 50;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  // Create a SoftmaxRegressionFunction. Regularization term ignored.
  SoftmaxRegressionFunction srf(data, labels, numClasses, 0);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double logLikelihood = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j)
    {
      arma::mat hypothesis, probabilities;

      hypothesis = arma::exp(parameters * data.col(j));
      probabilities = hypothesis / arma::accu(hypothesis);

      logLikelihood += log(probabilities(labels(j), 0));
    }
    logLikelihood /= points;

    // Compare with the value returned by the function.
    REQUIRE(srf.Evaluate(parameters) ==
        Approx(-logLikelihood).epsilon(1e-7));
  }
}

TEST_CASE("SoftmaxRegressionFunctionRegularizationEvaluate",
          "[SoftmaxRegressionTest]")
{
  const size_t points = 1000;
  const size_t trials = 50;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  // 3 objects for comparing regularization costs.
  SoftmaxRegressionFunction srfNoReg(data, labels, numClasses, 0);
  SoftmaxRegressionFunction srfSmallReg(data, labels, numClasses, 1);
  SoftmaxRegressionFunction srfBigReg(data, labels, numClasses, 20);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double wL2SquaredNorm;
    wL2SquaredNorm = arma::accu(parameters % parameters);

    // Calculate regularization terms.
    const double smallRegTerm = 0.5 * wL2SquaredNorm;
    const double bigRegTerm = 10 * wL2SquaredNorm;

    REQUIRE(srfNoReg.Evaluate(parameters) + smallRegTerm ==
        Approx(srfSmallReg.Evaluate(parameters)).epsilon(1e-7));
    REQUIRE(srfNoReg.Evaluate(parameters) + bigRegTerm ==
        Approx(srfBigReg.Evaluate(parameters)).epsilon(1e-7));
  }
}

TEST_CASE("SoftmaxRegressionFunctionGradient",
          "[SoftmaxRegressionTest]")
{
  const size_t points = 1000;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  // 2 objects for 2 terms in the cost function. Each term contributes towards
  // the gradient and thus need to be checked independently.
  SoftmaxRegressionFunction srf1(data, labels, numClasses, 0);
  SoftmaxRegressionFunction srf2(data, labels, numClasses, 20);

  // Create a random set of parameters.
  arma::mat parameters;
  parameters.randu(numClasses, inputSize);

  // Get gradients for the current parameters.
  arma::mat gradient1, gradient2;
  srf1.Gradient(parameters, gradient1);
  srf2.Gradient(parameters, gradient2);

  // Perturbation constant.
  const double epsilon = 0.0001;
  double costPlus1, costMinus1, numGradient1;
  double costPlus2, costMinus2, numGradient2;

  // For each parameter.
  for (size_t i = 0; i < numClasses; ++i)
  {
    for (size_t j = 0; j < inputSize; ++j)
    {
      // Perturb parameter with a positive constant and get costs.
      parameters(i, j) += epsilon;
      costPlus1 = srf1.Evaluate(parameters);
      costPlus2 = srf2.Evaluate(parameters);

      // Perturb parameter with a negative constant and get costs.
      parameters(i, j) -= 2 * epsilon;
      costMinus1 = srf1.Evaluate(parameters);
      costMinus2 = srf2.Evaluate(parameters);

      // Compute numerical gradients using the costs calculated above.
      numGradient1 = (costPlus1 - costMinus1) / (2 * epsilon);
      numGradient2 = (costPlus2 - costMinus2) / (2 * epsilon);

      // Restore the parameter value.
      parameters(i, j) += epsilon;

      // Compare numerical and backpropagation gradient values.
    REQUIRE(numGradient1 == Approx(gradient1(i, j)).epsilon(1e-4));
    REQUIRE(numGradient2 == Approx(gradient2(i, j)).epsilon(1e-4));
    }
  }
}

TEST_CASE("SoftmaxRegressionTwoClasses", "[SoftmaxRegressionTest]")
{
  const size_t points = 1000;
  const size_t inputSize = 3;
  const size_t numClasses = 2;
  const double lambda = 0.5;

  // Generate two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 2; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 2; i < points; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }

  // Train softmax regression object.
  SoftmaxRegression sr(data, labels, numClasses, lambda);

  // Compare training accuracy to 100.
  const double acc = sr.ComputeAccuracy(data, labels);
  REQUIRE(acc == Approx(100.0).epsilon(0.02));

  // Create test dataset.
  for (size_t i = 0; i < points / 2; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) =  0;
  }
  for (size_t i = points / 2; i < points; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }

  // Compare test accuracy to 100.
  const double testAcc = sr.ComputeAccuracy(data, labels);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.02));
}

TEST_CASE("SoftmaxRegressionFitIntercept", "[SoftmaxRegressionTest]")
{
  // Generate a two-Gaussian dataset,
  // which can't be separated without adding the intercept term.
  GaussianDistribution g1(arma::vec("1.0 1.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("9.0 9.0 9.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  SoftmaxRegression lr(data, responses, 2, 0.01, true);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.02));

  // Create a test set.
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = g1.Random();
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = g2.Random();
    responses[i] = 1;
  }

  // Ensure that the error is close to zero.
  const double testAcc = lr.ComputeAccuracy(data, responses);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.02));
}

TEST_CASE("SoftmaxRegressionMultipleClasses", "[SoftmaxRegressionTest]")
{
  const size_t points = 5000;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  arma::mat identity = arma::eye<arma::mat>(5, 5);
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train softmax regression object.
  SoftmaxRegression sr(data, labels, numClasses, lambda);

  // Compare training accuracy to 100.
  const double acc = sr.ComputeAccuracy(data, labels);
  REQUIRE(acc == Approx(100.0).epsilon(0.02));

  // Create test dataset.
  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Compare test accuracy to 100.
  const double testAcc = sr.ComputeAccuracy(data, labels);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.02));
}

TEST_CASE("SoftmaxRegressionTrainTest", "[SoftmaxRegressionTest]")
{
  // Test the stability of the SoftmaxRegression
  arma::mat dataset = arma::randu<arma::mat>(5, 1000);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 500; ++i)
    labels[i] = size_t(0.0);
  for (size_t i = 500; i < 1000; ++i)
    labels[i] = size_t(1.0);

  SoftmaxRegression sr(dataset.n_rows, 2);
  SoftmaxRegression sr2(dataset.n_rows, 2);
  sr.Parameters() = sr2.Parameters();
  ens::L_BFGS lbfgs;
  sr.Train(dataset, labels, 2, std::move(lbfgs));
  sr2.Train(dataset, labels, 2, std::move(lbfgs));

  // Ensure that the parameters are the same.
  REQUIRE(sr.Parameters().n_rows == sr2.Parameters().n_rows);
  REQUIRE(sr.Parameters().n_cols == sr2.Parameters().n_cols);
  for (size_t i = 0; i < sr.Parameters().n_elem; ++i)
  {
    if (std::abs(sr.Parameters()[i]) < 1e-4)
      REQUIRE(sr2.Parameters()[i] == Approx(0.0).margin(1e-4));
    else
        REQUIRE(sr.Parameters()[i] ==
            Approx(sr2.Parameters()[i]).epsilon(1e-6));
  }
}

TEST_CASE("SoftmaxRegressionOptimizerTrainTest", "[SoftmaxRegressionTest]")
{
  // The same as the previous test, just passing in an instantiated optimizer.
  arma::mat dataset = arma::randu<arma::mat>(5, 1000);
  arma::Row<size_t> labels(1000);
  for (size_t i = 0; i < 500; ++i)
    labels[i] = size_t(0.0);
  for (size_t i = 500; i < 1000; ++i)
    labels[i] = size_t(1.0);

  ens::L_BFGS lbfgs;
  SoftmaxRegression sr(dataset.n_rows, 2, true);

  ens::L_BFGS lbfgs2;
  SoftmaxRegression sr2(dataset.n_rows, 2, true);

  sr.Lambda() = sr2.Lambda() = 0.01;
  sr.Parameters() = sr2.Parameters();

  sr.Train(dataset, labels, 2, lbfgs);
  sr2.Train(dataset, labels, 2, lbfgs2);

  // Ensure that the parameters are the same.
  REQUIRE(sr.Parameters().n_rows == sr2.Parameters().n_rows);
  REQUIRE(sr.Parameters().n_cols == sr2.Parameters().n_cols);
  for (size_t i = 0; i < sr.Parameters().n_elem; ++i)
  {
    if (std::abs(sr.Parameters()[i]) < 1e-5)
      REQUIRE(sr2.Parameters()[i] == Approx(0.0).margin(1e-5));
    else
        REQUIRE(sr.Parameters()[i] ==
            Approx(sr2.Parameters()[i]).epsilon(1e-7));
  }
}

TEST_CASE("SoftmaxRegressionClassifySinglePointTest",
          "[SoftmaxRegressionTest]")
{
  const size_t points = 5000;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  arma::mat identity = arma::eye<arma::mat>(5, 5);
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train softmax regression object.
  SoftmaxRegression sr(data, labels, numClasses, lambda);

  // Create test dataset.
  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  sr.Classify(data, labels);

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    REQUIRE(sr.Classify(data.col(i)) == labels(i));
  }
}

TEST_CASE("SoftmaxRegressionComputeProbabilitiesTest",
          "[SoftmaxRegressionTest]")
{
  const size_t points = 5000;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  arma::mat identity = arma::eye<arma::mat>(5, 5);
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train softmax regression object.
  SoftmaxRegression sr(data, labels, numClasses, lambda);

  // Create test dataset.
  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  arma::mat probabilities;
  sr.Classify(data, probabilities);

  REQUIRE(probabilities.n_cols == data.n_cols);
  REQUIRE(probabilities.n_rows == sr.NumClasses());

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    REQUIRE(arma::sum(probabilities.col(i)) ==
        Approx(1.0).epsilon(1e-7));
  }
}

TEST_CASE("SoftmaxRegressionComputeProbabilitiesAndLabelsTest",
          "[SoftmaxRegressionTest]")
{
  const size_t points = 5000;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  arma::mat identity = arma::eye<arma::mat>(5, 5);
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

  arma::mat data(inputSize, points);
  arma::Row<size_t> labels(points);

  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train softmax regression object.
  SoftmaxRegression sr(data, labels, numClasses, lambda);

  // Create test dataset.
  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  arma::mat probabilities;
  arma::Row<size_t> testLabels;

  sr.Classify(data, labels);
  sr.Classify(data, testLabels, probabilities);

  REQUIRE(probabilities.n_cols == data.n_cols);
  REQUIRE(probabilities.n_rows == sr.NumClasses());

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    REQUIRE(arma::sum(probabilities.col(i)) ==
        Approx(1.0).epsilon(1e-7));
    REQUIRE(testLabels(i) == labels(i));
  }
}
