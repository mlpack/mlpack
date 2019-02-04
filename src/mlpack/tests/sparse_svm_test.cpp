/**
 * @file sparse_svm_test.cpp
 * @author Ayush Chamoli
 *
 * Test the Sparse SVM class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/sparse_svm/sparse_svm.hpp>
#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(SparseSVMTest);

/**
 * A simple test for SparseSVMFunction
 */
BOOST_AUTO_TEST_CASE(SparseSVMFunctionEvaluate)
{
  // A very simple fake dataset
  arma::mat dataset = "2 0 0;"
                      "0 0 0;"
                      "0 2 1;"
                      "1 0 2;"
                      "0 1 0";

  //  Corresponding labels
  arma::Row<size_t> labels = "1 0 1";

  SparseSVMFunction<arma::mat> svmf(dataset, labels, 2,
      0.0 /* no regularization */);

  // These were hand-calculated using Python.
  arma::mat parameters = "1 1 1 1 1;"
                         " 1 1 1 1 1";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 1.0, 1e-5);

  parameters = "2 0 1 2 2;"
               " 1 2 2 2 2";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 2.0, 1e-5);

  parameters = "-0.1425 8.3228 0.1724 -0.3374 0.1548;"
               "0.1435 0.0009 -0.1736 0.3356 -0.1544";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 0.0, 1e-5);

  parameters = "100 3 4 5 23;"
               "43 54 67 32 64";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 85.33333333, 1e-5);

  parameters = "3 71 22 12 6;"
               "100 39 30 57 22";
  BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), 11.0, 1e-5);
}

/**
 * A complicated test for the SparseSVMFunction for binary-class
 * classification.
 */
BOOST_AUTO_TEST_CASE(SparseSVMFunctionRandomBinaryEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 25;
  const size_t inputSize = 10;
  const size_t numClasses = 2;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // Create a SparseSVMFunction, Regularization term ignored.
  SparseSVMFunction<arma::mat> svmf(data, labels, numClasses,
      0.0 /* no regularization */);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i) {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    // Hand-calculate the loss function
    double hingeLoss = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j) {
      arma::mat score = parameters * data.col(j);
      double correct = score[labels(j)];
      for (size_t k = 0; k < numClasses; ++k) {
        if (k == labels[j])
          continue;
        double margin = score[k] - correct + 1;
        if (margin > 0)
          hingeLoss += margin;
      }
    }
    hingeLoss /= points;

    // Compare with the value returned by the function.
    BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), hingeLoss, 1e-5);
  }
}

/**
 * A complicated test for the SparseSVMFunction for multi-class
 * classification.
 */
BOOST_AUTO_TEST_CASE(SparseSVMFunctionRandomEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 25;
  const size_t inputSize = 10;
  const size_t numClasses = 5;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // Create a SparseSVMFunction, Regularization term ignored.
  SparseSVMFunction<arma::mat> svmf(data, labels, numClasses,
      0.0 /* no regularization */);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i) {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    // Hand-calculate the loss function
    double hingeLoss = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j) {
      arma::mat score = parameters * data.col(j);
      double correct = score[labels(j)];
      for (size_t k = 0; k < numClasses; ++k) {
        if (k == labels[j])
          continue;
        double margin = score[k] - correct + 1;
        if (margin > 0)
          hingeLoss += margin;
      }
    }
    hingeLoss /= points;

    // Compare with the value returned by the function.
    BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), hingeLoss, 1e-5);
  }
}

/**
 * Test regularization for the SparseSVMFunction Evaluate()
 * function.
 */
BOOST_AUTO_TEST_CASE(SparseSVMFunctionRegularizationEvaluate)
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
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // 3 objects for comparing regularization costs.
  SparseSVMFunction<arma::mat> svmfNoReg(data, labels, numClasses, 0);
  SparseSVMFunction<arma::mat> svmfSmallReg(data, labels, numClasses, 1);
  SparseSVMFunction<arma::mat> svmfBigReg(data, labels, numClasses, 20);

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

    BOOST_REQUIRE_CLOSE(svmfNoReg.Evaluate(parameters) + smallRegTerm,
                        svmfSmallReg.Evaluate(parameters), 1e-5);
    BOOST_REQUIRE_CLOSE(svmfNoReg.Evaluate(parameters) + bigRegTerm,
                        svmfBigReg.Evaluate(parameters), 1e-5);
  }
}

BOOST_AUTO_TEST_CASE(SparseSVMFunctionGradient)
{
  const size_t points = 1000;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // 2 objects for 2 terms in the cost function. Each term contributes towards
  // the gradient and thus need to be checked independently.
  SparseSVMFunction<arma::mat> svmf1(data, labels, numClasses, 0);
  SparseSVMFunction<arma::mat> svmf2(data, labels, numClasses, 10);

  // Create a random set of parameters.
  arma::mat parameters;
  parameters.randu(numClasses, inputSize);

  // Get gradients for the current parameters.
  arma::mat gradient1, gradient2;
  svmf1.Gradient(parameters, gradient1);
  svmf2.Gradient(parameters, gradient2);

  // Perturbation constant.
  const double epsilon = 0.001;
  double costPlus1, costMinus1, numGradient1;
  double costPlus2, costMinus2, numGradient2;


  // For each parameter.
  for (size_t i = 0; i < numClasses; i++)
  {
    for (size_t j = 0; j < inputSize; j++)
    {
      // Perturb parameter with a positive constant and get costs.
      parameters(i, j) += epsilon;
      costPlus1 = svmf1.Evaluate(parameters);
      costPlus2 = svmf2.Evaluate(parameters);

      // Perturb parameter with a negative constant and get costs.
      parameters(i, j) -= 2 * epsilon;
      costMinus1 = svmf1.Evaluate(parameters);
      costMinus2 = svmf2.Evaluate(parameters);

      // Compute numerical gradients using the costs calculated above.
      numGradient1 = (costPlus1 - costMinus1) / (2 * epsilon);
      numGradient2 = (costPlus2 - costMinus2) / (2 * epsilon);

      // Restore the parameter value.
      parameters(i, j) += epsilon;

      // Compare numerical and backpropagation gradient values.
      BOOST_REQUIRE_SMALL(numGradient1 - gradient1(i, j), 1e-2);
      BOOST_REQUIRE_SMALL(numGradient2 - gradient2(i, j), 1e-2);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
