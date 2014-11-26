/**
 * @file softmax_regression_test.cpp
 * @author Siddharth Agrawal
 *
 * Test the SoftmaxRegression class.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/softmax_regression/softmax_regression.hpp>

#include <boost/test/unit_test.hpp>
#include "old_boost_test_definitions.hpp"

using namespace mlpack;
using namespace mlpack::regression;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(SoftmaxRegressionTest);

BOOST_AUTO_TEST_CASE(SoftmaxRegressionFunctionEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 50;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::vec labels(points);
  for(size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // Create a SoftmaxRegressionFunction. Regularization term ignored.
  SoftmaxRegressionFunction srf(data, labels, inputSize, numClasses, 0);

  // Run a number of trials.
  for(size_t i = 0; i < trials; i++)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double logLikelihood = 0;

    // Compute error for each training example.
    for(size_t j = 0; j < points; j++)
    {
      arma::mat hypothesis, probabilities;

      hypothesis = arma::exp(parameters * data.col(j));
      probabilities = hypothesis / arma::accu(hypothesis);

      logLikelihood += log(probabilities(labels(j), 0));
    }
    logLikelihood /= points;

    // Compare with the value returned by the function.
    BOOST_REQUIRE_CLOSE(srf.Evaluate(parameters), -logLikelihood, 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SoftmaxRegressionFunctionRegularizationEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 50;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::vec labels(points);
  for(size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // 3 objects for comparing regularization costs.
  SoftmaxRegressionFunction srfNoReg(data, labels, inputSize, numClasses, 0);
  SoftmaxRegressionFunction srfSmallReg(data, labels, inputSize, numClasses, 1);
  SoftmaxRegressionFunction srfBigReg(data, labels, inputSize, numClasses, 20);

  // Run a number of trials.
  for (size_t i = 0; i < trials; i++)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double wL2SquaredNorm;
    wL2SquaredNorm = arma::accu(parameters % parameters);

    // Calculate regularization terms.
    const double smallRegTerm = 0.5 * wL2SquaredNorm;
    const double bigRegTerm = 10 * wL2SquaredNorm;

    BOOST_REQUIRE_CLOSE(srfNoReg.Evaluate(parameters) + smallRegTerm,
        srfSmallReg.Evaluate(parameters), 1e-5);
    BOOST_REQUIRE_CLOSE(srfNoReg.Evaluate(parameters) + bigRegTerm,
        srfBigReg.Evaluate(parameters), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SoftmaxRegressionFunctionGradient)
{
  const size_t points = 1000;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::vec labels(points);
  for(size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // 2 objects for 2 terms in the cost function. Each term contributes towards
  // the gradient and thus need to be checked independently.
  SoftmaxRegressionFunction srf1(data, labels, inputSize, numClasses, 0);
  SoftmaxRegressionFunction srf2(data, labels, inputSize, numClasses, 20);

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
  for (size_t i = 0; i < numClasses; i++)
  {
    for (size_t j = 0; j < inputSize; j++)
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
      BOOST_REQUIRE_CLOSE(numGradient1, gradient1(i, j), 1e-2);
      BOOST_REQUIRE_CLOSE(numGradient2, gradient2(i, j), 1e-2);
    }
  }
}

BOOST_AUTO_TEST_CASE(SoftmaxRegressionTwoClasses)
{
  const size_t points = 1000;
  const size_t inputSize = 3;
  const size_t numClasses = 2;
  const double lambda = 0.5;

  // Generate two-Gaussian dataset.
  GaussianDistribution g1(arma::vec("1.0 9.0 1.0"), arma::eye<arma::mat>(3, 3));
  GaussianDistribution g2(arma::vec("4.0 3.0 4.0"), arma::eye<arma::mat>(3, 3));

  arma::mat data(inputSize, points);
  arma::vec labels(points);

  for (size_t i = 0; i < points/2; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points/2; i < points; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }

  // Train softmax regression object.
  SoftmaxRegression<> sr(data, labels, inputSize, numClasses, lambda);

  // Compare training accuracy to 100.
  const double acc = sr.ComputeAccuracy(data, labels);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 0.5);

  // Create test dataset.
  for (size_t i = 0; i < points/2; i++)
  {
    data.col(i) = g1.Random();
    labels(i) =  0;
  }
  for (size_t i = points/2; i < points; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }

  // Compare test accuracy to 100.
  const double testAcc = sr.ComputeAccuracy(data, labels);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 0.6);
}

BOOST_AUTO_TEST_CASE(SoftmaxRegressionMultipleClasses)
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
  arma::vec labels(points);

  for (size_t i = 0; i < points/5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points/5; i < (2*points)/5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2*points)/5; i < (3*points)/5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3*points)/5; i < (4*points)/5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4*points)/5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Train softmax regression object.
  SoftmaxRegression<> sr(data, labels, inputSize, numClasses, lambda);

  // Compare training accuracy to 100.
  const double acc = sr.ComputeAccuracy(data, labels);
  BOOST_REQUIRE_CLOSE(acc, 100.0, 2.0);

  // Create test dataset.
  for (size_t i = 0; i < points/5; i++)
  {
    data.col(i) = g1.Random();
    labels(i) = 0;
  }
  for (size_t i = points/5; i < (2*points)/5; i++)
  {
    data.col(i) = g2.Random();
    labels(i) = 1;
  }
  for (size_t i = (2*points)/5; i < (3*points)/5; i++)
  {
    data.col(i) = g3.Random();
    labels(i) = 2;
  }
  for (size_t i = (3*points)/5; i < (4*points)/5; i++)
  {
    data.col(i) = g4.Random();
    labels(i) = 3;
  }
  for (size_t i = (4*points)/5; i < points; i++)
  {
    data.col(i) = g5.Random();
    labels(i) = 4;
  }

  // Compare test accuracy to 100.
  const double testAcc = sr.ComputeAccuracy(data, labels);
  BOOST_REQUIRE_CLOSE(testAcc, 100.0, 2.0);
}

BOOST_AUTO_TEST_SUITE_END();
