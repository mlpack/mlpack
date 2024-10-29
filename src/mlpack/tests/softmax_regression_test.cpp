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
  SoftmaxRegressionFunction<> srf(data, labels, numClasses, 0);

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

      hypothesis = exp(parameters * data.col(j));
      probabilities = hypothesis / accu(hypothesis);

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
  SoftmaxRegressionFunction<> srfNoReg(data, labels, numClasses, 0);
  SoftmaxRegressionFunction<> srfSmallReg(data, labels, numClasses, 1);
  SoftmaxRegressionFunction<> srfBigReg(data, labels, numClasses, 20);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double wL2SquaredNorm;
    wL2SquaredNorm = accu(parameters % parameters);

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
  SoftmaxRegressionFunction<> srf1(data, labels, numClasses, 0);
  SoftmaxRegressionFunction<> srf2(data, labels, numClasses, 20);

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
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0"),
      arma::eye<arma::mat>(3, 3));
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0"),
      arma::eye<arma::mat>(3, 3));

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
  SoftmaxRegression<> sr(data, labels, numClasses, lambda);

  // Compare training accuracy to 100.
  const double acc = sr.ComputeAccuracy(data, labels);
  REQUIRE(acc == Approx(100.0).epsilon(0.02));

  // Create test dataset.
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

  // Compare test accuracy to 100.
  const double testAcc = sr.ComputeAccuracy(data, labels);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.02));
}

TEMPLATE_TEST_CASE("SoftmaxRegressionFitIntercept", "[SoftmaxRegressionTest]",
    arma::fmat, arma::mat)
{
  using MatType = TestType;

  // Generate a two-Gaussian dataset,
  // which can't be separated without adding the intercept term.
  GaussianDistribution<> g1(arma::vec("1.0 1.0 1.0"),
      arma::eye<arma::mat>(3, 3));
  GaussianDistribution<> g2(arma::vec("9.0 9.0 9.0"),
      arma::eye<arma::mat>(3, 3));

  MatType data(3, 1000);
  arma::Row<size_t> responses(1000);
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g1.Random());
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g2.Random());
    responses[i] = 1;
  }

  // Now train a logistic regression object on it.
  SoftmaxRegression<MatType> lr(data, responses, 2, 0.01, true);

  // Ensure that the error is close to zero.
  const double acc = lr.ComputeAccuracy(data, responses);
  REQUIRE(acc == Approx(100.0).epsilon(0.02));

  // Create a test set.
  for (size_t i = 0; i < 500; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g1.Random());
    responses[i] = 0;
  }
  for (size_t i = 500; i < 1000; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g2.Random());
    responses[i] = 1;
  }

  // Ensure that the error is close to zero.
  const double testAcc = lr.ComputeAccuracy(data, responses);
  REQUIRE(testAcc == Approx(100.0).epsilon(0.02));
}

TEMPLATE_TEST_CASE("SoftmaxRegressionMultipleClasses",
    "[SoftmaxRegressionTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;
  using VecType = typename GetColType<TestType>::type;

  const size_t points = 5000;
  const size_t inputSize = 5;
  const size_t numClasses = 5;
  const double lambda = 0.5;

  // Generate five-Gaussian dataset.
  MatType identity = arma::eye<MatType>(5, 5);
  GaussianDistribution<MatType> g1(VecType("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution<MatType> g2(VecType("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution<MatType> g3(VecType("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution<MatType> g4(VecType("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution<MatType> g5(VecType("1.0 0.0 1.0 8.0 3.0"), identity);

  MatType data(inputSize, points);
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
  SoftmaxRegression<MatType> sr(data, labels, numClasses, lambda);

  // Compare training accuracy to 100.
  const double acc = sr.ComputeAccuracy(data, labels);
  REQUIRE(acc == Approx(100.0).epsilon(0.02));

  // Create test dataset.
  for (size_t i = 0; i < points / 5; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g1.Random());
    labels(i) = 0;
  }
  for (size_t i = points / 5; i < (2 * points) / 5; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g2.Random());
    labels(i) = 1;
  }
  for (size_t i = (2 * points) / 5; i < (3 * points) / 5; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g3.Random());
    labels(i) = 2;
  }
  for (size_t i = (3 * points) / 5; i < (4 * points) / 5; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g4.Random());
    labels(i) = 3;
  }
  for (size_t i = (4 * points) / 5; i < points; ++i)
  {
    data.col(i) = ConvTo<MatType>::From(g5.Random());
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

  SoftmaxRegression<> sr(dataset.n_rows, 2);
  SoftmaxRegression<> sr2(dataset.n_rows, 2);
  sr.Parameters() = sr2.Parameters();
  ens::L_BFGS lbfgs;
  sr.Train(dataset, labels, 2, lbfgs);
  sr2.Train(dataset, labels, 2, lbfgs);

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
  SoftmaxRegression<> sr(dataset.n_rows, 2, true);

  ens::L_BFGS lbfgs2;
  SoftmaxRegression<> sr2(dataset.n_rows, 2, true);

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
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution<> g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution<> g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution<> g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

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
  SoftmaxRegression<> sr(data, labels, numClasses, lambda);

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
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution<> g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution<> g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution<> g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

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
  SoftmaxRegression<> sr(data, labels, numClasses, lambda);

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

  arma::Row<size_t> predictions;
  arma::mat probabilities;
  sr.Classify(data, predictions, probabilities);

  REQUIRE(predictions.n_elem == data.n_cols);
  REQUIRE(probabilities.n_cols == data.n_cols);
  REQUIRE(probabilities.n_rows == sr.NumClasses());

  for (size_t i = 0; i < data.n_cols; ++i)
  {
    REQUIRE(predictions[i] < numClasses);
    REQUIRE(sum(probabilities.col(i)) == Approx(1.0).epsilon(1e-7));
  }

  // Test Classify() on a single point.
  size_t prediction = sr.Classify(data.col(0));
  REQUIRE(prediction == predictions[0]);

  arma::vec probabilitiesVec;
  sr.Classify(data.col(0), prediction, probabilitiesVec);
  REQUIRE(prediction == predictions[0]);
  REQUIRE(arma::approx_equal(probabilities.col(0), probabilitiesVec, "absdiff",
      1e-5));
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
  GaussianDistribution<> g1(arma::vec("1.0 9.0 1.0 2.0 2.0"), identity);
  GaussianDistribution<> g2(arma::vec("4.0 3.0 4.0 2.0 2.0"), identity);
  GaussianDistribution<> g3(arma::vec("3.0 2.0 7.0 0.0 5.0"), identity);
  GaussianDistribution<> g4(arma::vec("4.0 1.0 1.0 2.0 7.0"), identity);
  GaussianDistribution<> g5(arma::vec("1.0 0.0 1.0 8.0 3.0"), identity);

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
  SoftmaxRegression<> sr(data, labels, numClasses, lambda);

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
    REQUIRE(sum(probabilities.col(i)) == Approx(1.0).epsilon(1e-7));
    REQUIRE(testLabels(i) == labels(i));
  }
}

TEST_CASE("SoftmaxImmediateTrainTest", "[SoftmaxRegressionTest]")
{
  // Initialize a random dataset.
  const size_t numClasses = 3;
  const size_t inputSize = 10;
  const size_t points = 500;
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; ++i)
    labels(i) = RandInt(0, numClasses);

  // Train without setting any parameters to the constructor.
  SoftmaxRegression<> sr;
  sr.Train(data, labels, numClasses);

  // Now classify some points.
  // This just makes sure that the model can successfully make predictions at
  // all (i.e. no exception thrown).
  arma::Row<size_t> predictions;
  sr.Classify(data, predictions);

  REQUIRE(predictions.n_elem == labels.n_elem);
  REQUIRE(arma::all(predictions >= 0));
  REQUIRE(arma::all(predictions <= 2));
}

// Test variants of constructor.  This test is more about checking that all
// variants compile correctly than anything else.
TEMPLATE_TEST_CASE("SoftmaxRegressionConstructorVariantTest",
    "[SoftmaxRegressionTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;

  // Create random data.
  MatType data(50, 1000, arma::fill::randu);
  arma::Row<size_t> labels =
      arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 3));

  // Empty constructor.
  SoftmaxRegression<MatType> sr1;

  // No hyperparameters.
  SoftmaxRegression<MatType> sr2(data, labels, 4);

  // Specify hyperparameters only.
  SoftmaxRegression<MatType> sr3(data, labels, 4, 0.001, true);

  // Specify hyperparameters and a callback.
  SoftmaxRegression<MatType> sr4(data, labels, 4, 0.002, true,
      ens::EarlyStopAtMinLoss());

  // Specify hyperparameters and two callbacks.
  SoftmaxRegression<MatType> sr5(data, labels, 4, 0.003, true,
      ens::EarlyStopAtMinLoss(), ens::TimerStop(1000.0));

  // Specify hyperparameters and optimizer.
  ens::StandardSGD sgd(0.01);
  SoftmaxRegression<MatType> sr6(data, labels, 4, sgd, 0.004, true);

  // Specify hyperparameters, optimizer, and a callback.
  SoftmaxRegression<MatType> sr7(data, labels, 4, sgd, 0.005, true,
      ens::EarlyStopAtMinLoss());

  // Specify hyperparameters, optimizer, and two callbacks.
  SoftmaxRegression<MatType> sr8(data, labels, 4, sgd, 0.006, true,
      ens::EarlyStopAtMinLoss(), ens::TimerStop(1000.0));

  // Now we don't care what the training call actually produced, but we do care
  // that the model trained ate all and has the right size (except for the first
  // one).
  REQUIRE(sr1.Parameters().n_elem == 0);

  REQUIRE(sr2.Parameters().n_rows == 4);
  REQUIRE(sr2.Parameters().n_cols == 51);

  REQUIRE(sr3.Parameters().n_rows == 4);
  REQUIRE(sr3.Parameters().n_cols == 51);
  REQUIRE(sr3.Lambda() == Approx(0.001));
  REQUIRE(sr3.FitIntercept() == true);

  REQUIRE(sr4.Parameters().n_rows == 4);
  REQUIRE(sr4.Parameters().n_cols == 51);
  REQUIRE(sr4.Lambda() == Approx(0.002));
  REQUIRE(sr4.FitIntercept() == true);

  REQUIRE(sr5.Parameters().n_rows == 4);
  REQUIRE(sr5.Parameters().n_cols == 51);
  REQUIRE(sr5.Lambda() == Approx(0.003));
  REQUIRE(sr5.FitIntercept() == true);

  REQUIRE(sr6.Parameters().n_rows == 4);
  REQUIRE(sr6.Parameters().n_cols == 51);
  REQUIRE(sr6.Lambda() == Approx(0.004));
  REQUIRE(sr6.FitIntercept() == true);

  REQUIRE(sr7.Parameters().n_rows == 4);
  REQUIRE(sr7.Parameters().n_cols == 51);
  REQUIRE(sr7.Lambda() == Approx(0.005));
  REQUIRE(sr7.FitIntercept() == true);

  REQUIRE(sr8.Parameters().n_rows == 4);
  REQUIRE(sr8.Parameters().n_cols == 51);
  REQUIRE(sr8.Lambda() == Approx(0.006));
  REQUIRE(sr8.FitIntercept() == true);
}

// Test variants of Train().  This test is more about checking that all variants
// compile correctly than anything else.
TEMPLATE_TEST_CASE("SoftmaxRegressionTrainVariantTest",
    "[SoftmaxRegressionTest]", arma::fmat, arma::mat)
{
  using MatType = TestType;

  // Create random data.
  MatType data(50, 1000, arma::fill::randu);
  arma::Row<size_t> labels =
      arma::randi<arma::Row<size_t>>(1000, arma::distr_param(0, 3));

  // Create objects that we will use.
  SoftmaxRegression<MatType> sr1, sr2, sr3, sr4, sr5, sr6, sr7, sr8;

  // No hyperparameters.
  sr1.Train(data, labels, 4);

  // Specify hyperparameters only.
  sr2.Train(data, labels, 4, 0.001, false);

  // Specify hyperparameters and a callback.
  sr3.Train(data, labels, 4, 0.002, false, ens::EarlyStopAtMinLoss());

  // Specify hyperparameters and two callbacks.
  sr4.Train(data, labels, 4, 0.003, false, ens::EarlyStopAtMinLoss(),
      ens::TimerStop(1000.0));

  // Specify hyperparameters and an optimizer.
  ens::AdaDelta adaDelta(0.01);
  sr5.Train(data, labels, 4, adaDelta, 0.004, true);

  // Specify hyperparameters, an optimizer, and a callback.
  sr6.Train(data, labels, 4, adaDelta, 0.005, true, ens::EarlyStopAtMinLoss());

  // Specify hyperparameters, an optimizer, and two callbacks.
  sr7.Train(data, labels, 4, adaDelta, 0.006, true, ens::EarlyStopAtMinLoss(),
      ens::TimerStop(1000.0));

  // Now we don't care what the training actually produced, but we do want to
  // make sure that the model trained at all and has the right size.
  REQUIRE(sr1.Parameters().n_rows == 4);
  REQUIRE(sr1.Parameters().n_cols == 51);

  REQUIRE(sr2.Parameters().n_rows == 4);
  REQUIRE(sr2.Parameters().n_cols == 50);
  REQUIRE(sr2.Lambda() == Approx(0.001));
  REQUIRE(sr2.FitIntercept() == false);

  REQUIRE(sr3.Parameters().n_rows == 4);
  REQUIRE(sr3.Parameters().n_cols == 50);
  REQUIRE(sr3.Lambda() == Approx(0.002));
  REQUIRE(sr3.FitIntercept() == false);

  REQUIRE(sr4.Parameters().n_rows == 4);
  REQUIRE(sr4.Parameters().n_cols == 50);
  REQUIRE(sr4.Lambda() == Approx(0.003));
  REQUIRE(sr4.FitIntercept() == false);

  REQUIRE(sr5.Parameters().n_rows == 4);
  REQUIRE(sr5.Parameters().n_cols == 51);
  REQUIRE(sr5.Lambda() == Approx(0.004));
  REQUIRE(sr5.FitIntercept() == true);

  REQUIRE(sr6.Parameters().n_rows == 4);
  REQUIRE(sr6.Parameters().n_cols == 51);
  REQUIRE(sr6.Lambda() == Approx(0.005));
  REQUIRE(sr6.FitIntercept() == true);

  REQUIRE(sr7.Parameters().n_rows == 4);
  REQUIRE(sr7.Parameters().n_cols == 51);
  REQUIRE(sr7.Lambda() == Approx(0.006));
  REQUIRE(sr7.FitIntercept() == true);
}

// Make sure resetting a model does something.
TEST_CASE("SoftmaxRegressionResetTest", "[SoftmaxRegressionTest]")
{
  // Create random data.
  arma::mat data(50, 1000, arma::fill::randu);
  arma::Row<size_t> labels(1000);
  labels.subvec(500, 999).fill(1);

  // Create two logistic regression models.
  SoftmaxRegression<> sr1, sr2;

  // Initialize models to zeros.
  sr1.Parameters() = arma::zeros<arma::mat>(2, 51);
  sr2.Parameters() = arma::zeros<arma::mat>(2, 51);

  ens::L_BFGS lbfgs1(1, 1); // Use only one iteration.
  ens::L_BFGS lbfgs2(1, 1);

  sr1.Train(data, labels, 2, lbfgs1);
  sr2.Train(data, labels, 2, lbfgs2);

  REQUIRE(
      arma::approx_equal(sr1.Parameters(), sr2.Parameters(), "absdiff", 1e-5));

  // Now reset one model and incrementally train the other.
  sr1.Reset();
  sr1.Train(data, labels, 2, lbfgs1);
  sr2.Train(data, labels, 2, lbfgs2);

  REQUIRE(
      !arma::approx_equal(sr1.Parameters(), sr2.Parameters(), "absdiff", 1e-5));
}
