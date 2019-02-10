/**
 * @file linear_svm_test.cpp
 * @author Ayush Chamoli
 *
 * Test the Linear SVM class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <mlpack/methods/linear_svm/linear_svm.hpp>
#include <ensmallen.hpp>

#include <boost/test/unit_test.hpp>
#include "test_tools.hpp"

using namespace mlpack;
using namespace mlpack::svm;
using namespace mlpack::distribution;

BOOST_AUTO_TEST_SUITE(LinearSVMTest);

/**
 * A simple test for LinearSVMFunction
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionEvaluate)
{
  // A very simple fake dataset
  arma::mat dataset = "2 0 0;"
                      "0 0 0;"
                      "0 2 1;"
                      "1 0 2;"
                      "0 1 0";

  //  Corresponding labels
  arma::Row<size_t> labels = "1 0 1";

  LinearSVMFunction<arma::mat> svmf(dataset, labels, 2,
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
 * A complicated test for the LinearSVMFunction for binary-class
 * classification.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionRandomBinaryEvaluate)
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

  // Create a LinearSVMFunction, Regularization term ignored.
  LinearSVMFunction<arma::mat> svmf(data, labels, numClasses,
      0.0 /* no regularization */);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    // Hand-calculate the loss function
    double hingeLoss = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j)
    {
      arma::mat score = parameters * data.col(j);
      double correct = score[labels(j)];
      for (size_t k = 0; k < numClasses; ++k)
      {
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
 * A complicated test for the LinearSVMFunction for multi-class
 * classification.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionRandomEvaluate)
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

  // Create a LinearSVMFunction, Regularization term ignored.
  LinearSVMFunction<arma::mat> svmf(data, labels, numClasses,
      0.0 /* no regularization */);

  // Run a number of trials.
  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    // Hand-calculate the loss function
    double hingeLoss = 0;

    // Compute error for each training example.
    for (size_t j = 0; j < points; ++j)
    {
      arma::mat score = parameters * data.col(j);
      double correct = score[labels(j)];
      for (size_t k = 0; k < numClasses; ++k)
      {
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
 * Test regularization for the LinearSVMFunction Evaluate()
 * function.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionRegularizationEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 25;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  // 3 objects for comparing regularization costs.
  LinearSVMFunction<arma::mat> svmfNoReg(data, labels, numClasses, 0);
  LinearSVMFunction<arma::mat> svmfSmallReg(data, labels, numClasses, 1);
  LinearSVMFunction<arma::mat> svmfBigReg(data, labels, numClasses, 20);

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

/**
 * Test individual Evaluate() functions to be used for
 * optimization.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionSeparableEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 25;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  LinearSVMFunction<> svmf(data, labels, numClasses);

  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double hingeLoss = 0;
    for (size_t j = 0; j < points; ++j)
      hingeLoss += svmf.Evaluate(parameters, j, 1);

    hingeLoss /= points;

    // Compare with the value returned by the function.
    BOOST_REQUIRE_CLOSE(svmf.Evaluate(parameters), hingeLoss, 1e-5);
  }
}

/**
 *
 * Test regularization for the separable Evaluate() function
 * to be used Optimizers.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionRegularizationSeparableEvaluate)
{
  const size_t points = 1000;
  const size_t trials = 25;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  LinearSVMFunction<> svmfNoReg(data, labels, numClasses, 0.0);
  LinearSVMFunction<> svmfSmallReg(data, labels, numClasses, 0.5);
  LinearSVMFunction<> svmfBigReg(data, labels, numClasses, 20.0);


  // Check that the number of functions is correct.
  BOOST_REQUIRE_EQUAL(svmfNoReg.NumFunctions(), points);
  BOOST_REQUIRE_EQUAL(svmfSmallReg.NumFunctions(), points);
  BOOST_REQUIRE_EQUAL(svmfBigReg.NumFunctions(), points);


  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    double wL2SquaredNorm;
    wL2SquaredNorm = 0.5 * arma::accu(parameters % parameters);

    // Calculate regularization terms.
    const double smallRegTerm = 0.5 * wL2SquaredNorm;
    const double bigRegTerm = 20 * wL2SquaredNorm;

    for (size_t j = 0; j < points; ++j)
    {
      BOOST_REQUIRE_CLOSE(svmfNoReg.Evaluate(parameters, j, 1) + smallRegTerm,
          svmfSmallReg.Evaluate(parameters, j, 1), 1e-5);
      BOOST_REQUIRE_CLOSE(svmfNoReg.Evaluate(parameters, j, 1) + bigRegTerm,
          svmfBigReg.Evaluate(parameters, j, 1), 1e-5);
    }
  }
}

/**
 * Test gradient for the LinearSVMFunction.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionGradient)
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
  LinearSVMFunction<arma::mat> svmf1(data, labels, numClasses, 0.0);
  LinearSVMFunction<arma::mat> svmf2(data, labels, numClasses, 10.0);

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
      BOOST_REQUIRE_SMALL(numGradient1 - gradient1(i, j), 1e-5);
      BOOST_REQUIRE_SMALL(numGradient2 - gradient2(i, j), 1e-5);
    }
  }
}

/**
 * Test separable Gradient() of the LinearSVMFunction when regularization
 * is used.
 */
BOOST_AUTO_TEST_CASE(LinearSVMFunctionSeparableGradient)
{
  const size_t points = 1000;
  const size_t trials = 3;
  const size_t inputSize = 10;
  const size_t numClasses = 3;

  // Initialize a random dataset.
  arma::mat data;
  data.randu(inputSize, points);

  // Create random class labels.
  arma::Row<size_t> labels(points);
  for (size_t i = 0; i < points; i++)
    labels(i) = math::RandInt(0, numClasses);

  LinearSVMFunction<> svmfNoReg(data, labels, numClasses, 0.0);
  LinearSVMFunction<> svmfSmallReg(data, labels, numClasses, 0.5);
  LinearSVMFunction<> svmfBigReg(data, labels, numClasses, 20.0);

  for (size_t i = 0; i < trials; ++i)
  {
    // Create a random set of parameters.
    arma::mat parameters;
    parameters.randu(numClasses, inputSize);

    arma::mat gradient;
    arma::mat smallRegGradient;
    arma::mat bigRegGradient;

    // Test separable gradient for each point.  Regularization will be the same.
    for (size_t k = 0; k < points; ++k)
    {
      svmfNoReg.Gradient(parameters, k, gradient, 1);
      svmfSmallReg.Gradient(parameters, k, smallRegGradient, 1);
      svmfBigReg.Gradient(parameters, k, bigRegGradient, 1);

      // Check sizes of gradients.
      BOOST_REQUIRE_EQUAL(gradient.n_elem, parameters.n_elem);
      BOOST_REQUIRE_EQUAL(smallRegGradient.n_elem, parameters.n_elem);
      BOOST_REQUIRE_EQUAL(bigRegGradient.n_elem, parameters.n_elem);

      // Check other terms.
      for (size_t j = 0; j < parameters.n_elem; ++j)
      {
        const double smallRegTerm = 0.5 * parameters[j];
        const double bigRegTerm = 20.0 * parameters[j];

        BOOST_REQUIRE_CLOSE(gradient[j] + smallRegTerm, smallRegGradient[j],
                            1e-5);
        BOOST_REQUIRE_CLOSE(gradient[j] + bigRegTerm, bigRegGradient[j], 1e-5);
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END();
